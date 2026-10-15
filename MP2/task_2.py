import jsonlines
import sys
import os
import ast
import json
import shutil
import subprocess
import importlib.util
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

MAX_NEW_TOKENS = 1024
PYTEST_TIMEOUT = 300  # seconds, guards against generated tests that never finish

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

#####################################################
# Prompt construction
#####################################################

# Vanilla prompt: the instruction given in the MP2 description, unchanged.
VANILLA_INSTRUCTION = """Generate a pytest test suite for the following code.
Only write unit tests in the output and nothing else.
{program}"""

# Crafted prompt: explicit coverage goal, path enumeration before writing tests,
# one assert per test (so pass/fail can be counted per test), oracle guidance,
# and a one-shot example of the expected output format.
CRAFTED_INSTRUCTION = """Write a pytest test suite for the Python function `{entry_point}` shown below. The function is saved in the module `{module}`.

Goal: reach 100% statement coverage AND 100% branch coverage, with every test passing.

Follow these steps:
1. Read the code and list every execution path as Python comments at the top of the test file: both the true and the false side of every if/elif/else, every loop executed zero times, once, and several times, every early return, and every exception the code raises explicitly.
2. Write at least one test for each path in your list. Also turn every example from the docstring into a test.
3. For each test, compute the expected value by tracing THIS code line by line on that input. The code is always correct. If you are not sure what the code returns for an input, choose a simpler input.

Rules:
- Start the file with `import pytest` and `from {module} import {entry_point}`.
- Do NOT copy or re-implement the function under test.
- Write between 8 and 15 test functions. Name each one `test_<what_it_checks>` and put exactly ONE assert in each.
- Only use inputs that are valid according to the docstring, unless the code explicitly handles invalid input.
- Use `pytest.approx` when comparing floats.
- Output a single ```python code block and nothing else.

Example. For this code in module `example_module`:
```python
def classify(n):
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    for d in range(2, n):
        if n % d == 0:
            return "composite"
    return "prime-or-one"
```
a good answer is:
```python
import pytest
from example_module import classify

# Paths:
# 1. n < 0 -> "negative"
# 2. n == 0 -> "zero"
# 3. loop runs zero times (n = 1 or n = 2) -> "prime-or-one"
# 4. loop finds a divisor -> "composite"
# 5. loop runs several times without finding a divisor -> "prime-or-one"

def test_negative():
    assert classify(-5) == "negative"

def test_zero():
    assert classify(0) == "zero"

def test_one_loop_not_entered():
    assert classify(1) == "prime-or-one"

def test_two_loop_not_entered():
    assert classify(2) == "prime-or-one"

def test_composite_found_on_first_divisor():
    assert classify(4) == "composite"

def test_composite_found_after_several_iterations():
    assert classify(9) == "composite"

def test_prime_full_loop():
    assert classify(7) == "prime-or-one"

def test_larger_prime():
    assert classify(13) == "prime-or-one"
```

Now write the test suite for this code in module `{module}`:
```python
{program}
```"""

def build_instruction(entry, module, program, vanilla):
    # str.replace instead of str.format: the programs may contain braces
    if vanilla:
        return VANILLA_INSTRUCTION.replace("{program}", program)
    return (CRAFTED_INSTRUCTION
            .replace("{entry_point}", entry["entry_point"])
            .replace("{module}", module)
            .replace("{program}", program))

#####################################################
# Response processing
#####################################################

def extract_code(response):
    """Return the first fenced code block of the response (or the whole response)."""
    for fence in ("```python", "```py", "```"):
        if fence in response:
            code = response.split(fence, 1)[1]
            # A missing closing fence means the answer was cut off; keep what we have
            return code.split("```", 1)[0]
    return response

SEGMENT_STARTS = ("def ", "async def ", "class ", "@", "import ", "from ", "if __name__")

def drop_unparsable_segments(code):
    """
    Keep the test file runnable when the model produced a syntax error or was truncated:
    split the code into top-level segments (imports, decorated functions, classes) and
    drop the segments that do not parse on their own.
    Returns (code, number_of_dropped_segments).
    """
    try:
        ast.parse(code)
        return code, 0
    except SyntaxError:
        pass

    segments = []
    current = []
    for line in code.splitlines():
        starts_segment = line.startswith(SEGMENT_STARTS)
        only_decorators = current and all(l.startswith("@") or not l.strip() for l in current)
        if starts_segment and current and not only_decorators:
            segments.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        segments.append("\n".join(current))

    kept, dropped = [], 0
    for segment in segments:
        try:
            ast.parse(segment)
            kept.append(segment)
        except SyntaxError:
            dropped += 1
    return "\n\n".join(kept), dropped

def is_main_guard(node):
    return (isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__")

def module_exists(name, module):
    return name == module or importlib.util.find_spec(name) is not None

def build_test_file(code, module, program):
    """
    Turn the extracted code into a test file that imports the function under test:
    - drop imports of modules that do not exist (e.g. `from solution import f`)
    - drop copies of the functions under test, so the tests exercise {module}.py
    - drop `if __name__ == "__main__"` blocks and bare top-level calls
    - always import everything from {module}
    """
    program_functions = {
        node.name for node in ast.parse(program).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    body = []
    for node in ast.parse(code).body:
        if isinstance(node, ast.Import):
            node.names = [a for a in node.names if module_exists(a.name.split(".")[0], module)]
            if not node.names:
                continue
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module or not module_exists(node.module.split(".")[0], module):
                continue
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in program_functions:
            continue
        elif is_main_guard(node):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            continue
        body.append(node)

    header = f"import pytest\nfrom {module} import *\n\n"
    return header + ast.unparse(ast.Module(body=body, type_ignores=[])) + "\n"

#####################################################
# Running the tests
#####################################################

def read_junit(junit_path):
    """Return (total, passed, failed) from a pytest JUnit XML report."""
    if not os.path.exists(junit_path):
        return 0, 0, 0
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    total = failed = skipped = 0
    for suite in suites:
        total += int(suite.get("tests", 0))
        failed += int(suite.get("failures", 0)) + int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
    return total, total - failed - skipped, failed

def run_pytest(module, test_file, work_dir, cov_path, junit_path):
    """
    Equivalent of `pytest {ID}_test.py --cov {ID} --cov-branch --cov-report json:Coverage/...`.
    Returns the pytest output.
    """
    command = [
        sys.executable, "-m", "pytest", test_file,
        "--cov", module, "--cov-branch",
        "--cov-report", f"json:{cov_path}",
        "--junitxml", junit_path,
        "-p", "no:cacheprovider",
    ]
    try:
        result = subprocess.run(command, cwd=work_dir, capture_output=True, text=True, timeout=PYTEST_TIMEOUT)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"pytest timed out after {PYTEST_TIMEOUT} seconds"

def read_coverage(cov_path):
    with open(cov_path, "r") as file:
        totals = json.load(file)["totals"]
    line_coverage = 100.0 * totals["covered_lines"] / totals["num_statements"] if totals["num_statements"] else 100.0
    branch_coverage = 100.0 * totals["covered_branches"] / totals["num_branches"] if totals["num_branches"] else 100.0
    return totals["percent_covered"], line_coverage, branch_coverage

def evaluate_tests(module, program, test_code, prompt_type):
    """Write {ID}.py and {ID}_test.py, run pytest with coverage, and collect the results."""
    work_dir = os.path.abspath(f"generated_tests_{prompt_type}")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs("Coverage", exist_ok=True)

    test_file = f"{module}_test.py"
    save_file(program, os.path.join(work_dir, f"{module}.py"))
    save_file(test_code, os.path.join(work_dir, test_file))

    cov_path = os.path.abspath(os.path.join("Coverage", f"{module}_test_{prompt_type}.json"))
    junit_path = os.path.join(work_dir, f"{module}_junit.xml")
    for stale in (cov_path, junit_path):
        if os.path.exists(stale):
            os.remove(stale)

    pytest_output = run_pytest(module, test_file, work_dir, cov_path, junit_path)
    total, passed, failed = read_junit(junit_path)

    collection_error = not os.path.exists(cov_path)
    if collection_error:
        # The suite could not run at all (import error, timeout, ...). Measure what the
        # import of the module alone covers so that a coverage report still exists.
        stub_file = f"{module}_stub_test.py"
        save_file(f"from {module} import *\n\ndef test_import():\n    pass\n",
                  os.path.join(work_dir, stub_file))
        run_pytest(module, stub_file, work_dir, cov_path, os.path.join(work_dir, "stub_junit.xml"))

    coverage, line_coverage, branch_coverage = read_coverage(cov_path)
    return {
        "coverage": coverage,
        "line_coverage": line_coverage,
        "branch_coverage": branch_coverage,
        "num_tests": total,
        "num_passed": passed,
        "num_failed": failed,
        "collection_error": collection_error,
        "pytest_output": pytest_output,
    }

#####################################################
# Main loop
#####################################################

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")

    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # TODO: load the model with quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    prompt_type = "vanilla" if vanilla else "crafted"
    # Start from a clean directory so files from an earlier run cannot leak into this one
    shutil.rmtree(f"generated_tests_{prompt_type}", ignore_errors=True)

    results = []
    for entry in dataset:
        module = entry["task_id"].replace("/", "_")
        program = entry["prompt"] + entry["canonical_solution"]

        # TODO: create prompt for the model
        # The DeepSeek chat template adds the system prompt and the
        # "### Instruction:" / "### Response:" markers around the instruction.
        messages = [{"role": "user", "content": build_instruction(entry, module, program, vanilla)}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # TODO: prompt the model and get the response
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # TODO: process the response, generate coverage and save it to results
        code, dropped_segments = drop_unparsable_segments(extract_code(response))
        test_code = build_test_file(code, module, program)
        evaluation = evaluate_tests(module, program, test_code, prompt_type)
        coverage = evaluation["coverage"]

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}")
        print(f"test file ({dropped_segments} unparsable segment(s) dropped):\n{test_code}")
        print(f"pytest output:\n{evaluation['pytest_output']}")
        print(f"tests: {evaluation['num_tests']}, passed: {evaluation['num_passed']}, failed: {evaluation['num_failed']}, "
              f"collection_error: {evaluation['collection_error']}")
        print(f"line coverage: {evaluation['line_coverage']:.2f}, branch coverage: {evaluation['branch_coverage']:.2f}")
        print(f"coverage:\n{coverage}\n")

        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "coverage": coverage,
            "line_coverage": evaluation["line_coverage"],
            "branch_coverage": evaluation["branch_coverage"],
            "num_tests": evaluation["num_tests"],
            "num_passed": evaluation["num_passed"],
            "num_failed": evaluation["num_failed"],
            "collection_error": evaluation["collection_error"],
            "dropped_segments": dropped_segments,
            "test_code": test_code,
        })

    print_summary(results, prompt_type)
    return results

def print_summary(results, prompt_type):
    n = len(results)
    print(f"===== Summary ({prompt_type}) =====")
    print(f"{'task_id':<16}{'tests':>7}{'passed':>8}{'failed':>8}{'line%':>9}{'branch%':>9}{'coverage%':>11}")
    for r in results:
        print(f"{r['task_id']:<16}{r['num_tests']:>7}{r['num_passed']:>8}{r['num_failed']:>8}"
              f"{r['line_coverage']:>9.2f}{r['branch_coverage']:>9.2f}{r['coverage']:>11.2f}")
    total = sum(r["num_tests"] for r in results)
    passed = sum(r["num_passed"] for r in results)
    print(f"Average coverage: {sum(r['coverage'] for r in results) / n:.2f}")
    print(f"Average line coverage: {sum(r['line_coverage'] for r in results) / n:.2f}")
    print(f"Average branch coverage: {sum(r['branch_coverage'] for r in results) / n:.2f}")
    print(f"Tests: {total}, passed: {passed} ({100.0 * passed / total if total else 0:.2f}%), failed: {total - passed}")

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 task_2.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt

    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3] # True or False

    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")

    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")

    vanilla = True if if_vanilla == "True" else False

    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)

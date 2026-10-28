import jsonlines
import sys
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)
def extract_java_code(response):
    """Extract the requested tags, with Markdown code fences as a fallback."""
    match = re.search(r"\[Java Start\](.*?)\[Java End\]", response, re.DOTALL)
    code = match.group(1).strip() if match else response.strip()
    fences = re.findall(r"```(?:java)?\s*\n(.*?)```", code, re.DOTALL | re.IGNORECASE)
    if fences:
        return "\n\n".join(fences).strip()
    return code if match else ""


def evaluate_translation(response, java_entry):
    """Return correctness, failure category and compiler/test diagnostics."""
    code = extract_java_code(response)
    if not code:
        return False, "extraction_error", "No tagged or fenced Java code found."
    imports = re.findall(r"^\s*import\s+[^;]+;", code, re.MULTILINE)
    code = re.sub(r"^\s*import\s+[^;]+;", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*package\s+[^;]+;", "", code, flags=re.MULTILINE)
    if not re.search(r"\bclass\s+Solution\b", code) and re.search(r"\bclass\s+Main\b", code):
        # Rename the generated class and its references before appending the
        # dataset's Main test driver. Its example main method is never executed.
        code = re.sub(r"\bMain\b", "Solution", code)
    if re.search(r"\bclass\s+Solution\b", code):
        code = re.sub(r"\bpublic\s+class\s+Solution\b", "class Solution", code)
    elif not re.search(r"\bclass\s+\w+", code):
        code = "class Solution {\n" + code + "\n}"
    else:
        return False, "extraction_error", "Expected Solution class or standalone method."
    source = "\n".join(["import java.util.*;", "import java.lang.*;", *imports,
                         code, java_entry["test"]])
    with tempfile.TemporaryDirectory(prefix="mp3_java_") as directory:
        Path(directory, "Main.java").write_text(source, encoding="utf-8")
        for status, command in [
            ("compilation_error", ["javac", "-encoding", "UTF-8", "Main.java"]),
            ("test_failure", ["java", "-ea", "-cp", directory, "Main"]),
        ]:
            try:
                result = subprocess.run(command, cwd=directory, capture_output=True,
                                        text=True, timeout=30)
            except subprocess.TimeoutExpired:
                return False, status + "_timeout", "Exceeded 30 seconds."
            if result.returncode != 0:
                return False, status, result.stdout + result.stderr
    return True, "passed", ""


def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True,
                 java_dataset=None):
    print(f"Working with {model_name} prompt type {vanilla}...")
    if java_dataset is None:
        raise ValueError("The matching Java dataset is required for evaluation.")
    for executable in ("javac", "java"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Missing {executable}: install a JDK before running Task 1.")
    java_by_id = {item["task_id"].split("/")[-1]: item for item in java_dataset}
    for entry in dataset:
        if entry["task_id"].split("/")[-1] not in java_by_id:
            raise ValueError(f"Missing Java tests for {entry['task_id']}")

    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True
    )
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
    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create
        #       the prompt including prompt, canonical_solution, test, etc.
        if vanilla:
            python_code = entry.get("declaration", entry["prompt"]) + entry["canonical_solution"]
            prompt = f"""You are an AI programming assistant. Can you translate the following Python code into Java?
{python_code}

        The new Java code must be enclosed between [Java Start] and [Java End] tag,

        ### Response:
        """
        else:
            python_code = entry.get("declaration", entry["prompt"]) + entry["canonical_solution"]
            java_entry = java_by_id[entry["task_id"].split("/")[-1]]
            prompt = f"""Translate the following Python code into Java.

Python code:
{python_code}

Required Java declaration:
{java_entry['declaration']}

Requirements:
1. Use the exact class name, method name, parameter types, and return type in the Java declaration.
2. Preserve the Python code's behavior, including boundary conditions, division, modulo, and collection operations.
3. Return a complete, compilable Solution class with all necessary imports and closing braces. Do not include a main method.
4. Output only Java code enclosed between [Java Start] and [Java End], with no explanation.
"""
        # TODO: prompt the model and get the response
        messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, input_length:]

        response = tokenizer.decode(new_tokens, skip_special_tokens=True)


        # TODO: process the response and save it to results
        java_entry = java_by_id[entry["task_id"].split("/")[-1]]
        verdict, status, diagnostics = evaluate_translation(response, java_entry)
        print(f"Validation: {status}\n{diagnostics}")

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
        })

    return results
# def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
#     print(f"Working with {model_name} prompt type {vanilla}...")

#     # TODO: download the model
#     # TODO: load the model with quantization

#     results = []
#     for entry in dataset:
#         # TODO: create prompt for the model
#         # Tip : Use can use any data from the dataset to create
#         #       the prompt including prompt, canonical_solution, test, etc.
#         prompt = ""

#         # TODO: prompt the model and get the response
#         response = ""

#         # TODO: process the response and save it to results
#         verdict = False

#         print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
#         results.append({
#             "task_id": entry["task_id"],
#             "prompt": prompt,
#             "response": response,
#             "is_correct": verdict
#         })

#     return results

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
    This Python script is to run prompt LLMs for code translation.
    Usage:
    `python3 task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
    # An optional fifth argument overrides the inferred Java dataset path.
    java_path = args[4] if len(args) > 4 else input_dataset.replace(
        "selected_humanevalx_python_", "selected_humanevalx_java_"
    )
    if java_path == input_dataset:
        raise ValueError("Pass the Java dataset path as the fifth argument.")
    java_dataset = read_jsonl(java_path)
    results = prompt_model(dataset, model, vanilla, java_dataset)
    write_jsonl(results, output_file)

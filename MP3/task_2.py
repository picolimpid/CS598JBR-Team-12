import jsonlines
import sys
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def build_instruction(entry, module, program, vanilla):
    if vanilla:
        return VANILLA_INSTRUCTION.replace("{program}", program)
    return (CRAFTED_INSTRUCTION
            .replace("{entry_point}", entry["entry_point"])
            .replace("{module}", module)
            .replace("{program}", program))

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

    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        python_code = entry.get("declaration", entry["prompt"]) + entry["buggy_solution"]
        prompt = f"""You are an AI programming assistant. Can you determine the correctness of the following python code?
```py
{python_code}
```

Is the above code buggy or correct?
Explain your step by step reasoning. Your prediction should be enclosed within <start> and <end> tags.
Reply with either `<start>buggy<end>` or `<start>correct<end>`.
Your response will be rejected if you don't include your prediction in tags at the end of your response.
Do not try and fix the snippet if it's buggy, simply answer whether it is buggy or correct.
"""
        if not vanilla:
            prompt = f"""You are an expert Python code reviewer.

Determine whether the implementation satisfies the ORIGINAL
function specification. A program is buggy if there exists
at least one valid input for which its behavior differs
from the specification.

### Original Specification
{entry["prompt"]}

### Implementation to Review
```python
{python_code}
```

### Bug Information
- Suspected bug category: {entry["bug_type"]}
- Possible failure symptom: {entry["failure_symptoms"]}

### Review Instructions
1. Identify what the specification requires, including
   examples, return values, and edge cases.
2. Trace the implementation carefully. Pay special attention
   to operators, variable usage, return statements, loop
   conditions, and Python built-in function behavior.
3. Test the specification's examples mentally. Try to find
   a valid counterexample showing a mismatch.
4. If a valid counterexample exists, the code is buggy.
   If no mismatch can be identified, classify it as correct.
   Do not rely solely on the suspected bug category.

### Required Output
Briefly state the expected behavior, actual behavior, and
any concrete counterexample you found.
Limit your explanation to 80 words.

Then write your FINAL prediction on a separate last line,
using exactly one tag:
<start>buggy<end> or <start>correct<end>

Do not output a prediction tag before the final line.
Do not output both tags.
Do not revise your prediction after writing it.
Do not rewrite or fix the code.
"""
        # print(prompt)
        messages = [{ "role": "user", "content": prompt }]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, input_length:]

        # TODO: prompt the model and get the response
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # print(response)

        # TODO: process the response and save it to results
        match = re.search(r"<start>(.*?)<end>", response)
        if match:
            llm_answer = match.group(1).lower()
        else:
            llm_answer = 'correct' if 'correct' in response else 'buggy'
        verdict = llm_answer == 'buggy'

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
        })

    return results

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
    This Python script is to run prompt LLMs for bug detection.
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


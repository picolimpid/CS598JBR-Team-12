import jsonlines
import sys
import torch
import ast
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def clean_predicted_output(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    return lines[-1]

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    random.seed(42)
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
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

    model.eval()
    
    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        program = entry["prompt"] + entry["canonical_solution"]

        test_tree = ast.parse(entry["test"])

        assertions = [
            node
            for node in ast.walk(test_tree)
            if isinstance(node, ast.Assert)
        ]

        selected_assertion = random.choice(assertions)

        comparison = selected_assertion.test
        candidate_call = comparison.left
        expected_node = comparison.comparators[0]

        input_args = ", ".join(
            ast.unparse(arg)
            for arg in candidate_call.args
        )

        expected_output = ast.unparse(expected_node)

        # print(entry["task_id"], input_args, expected_output)

        if vanilla:
            prompt = f"""You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:
If the input is {input_args}, what will the following code return?

The return value prediction must be enclosed between [Output] and [/Output] tags. For example: [Output]prediction[/Output].

{program}

### Response:
"""
        else:
            function_call = f"{entry['entry_point']}({input_args})"

            prompt = f"""Predict the exact return value of a Python function call.

Example:
Function call:
example("hello")

Code:
def example(x):
    return x

Response:
[Output]'hello'[/Output]

Now solve this problem.

Code:
{program}

Rules:
- The docstring may contain examples. Those examples are documentation only.
- Never use, copy, or substitute any input or output from a docstring example.
- Evaluate ONLY the exact function call shown below.
- Execute the actual Python code mentally and determine its returned value.
- Preserve the exact Python return type and representation.
- Strings must include quotes.
- Your entire response must be exactly one line.
- Do not explain anything.
- Do not output code or the function call itself.

Function call to evaluate:
{function_call}

Response format:
[Output]<return_value>[/Output]

Response:
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
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, input_length:]

        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # TODO: process the response and save it to results
        matches = re.findall(r"\[Output\](.*?)\[/Output\]", response, re.DOTALL)

        if matches:
            predicted_output = clean_predicted_output(matches[-1])

            expected_value = expected_output.strip()
            predicted_value = predicted_output.strip()

            verdict = predicted_value == expected_value
        else:
            predicted_output = ""
            verdict = False

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
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
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 Task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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

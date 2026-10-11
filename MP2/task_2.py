import jsonlines
import sys
import torch
import subprocess
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
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

    prompt_type = "vanilla" if vanilla else "crafted"

    results = []
    for entry in dataset:
        ID = entry["task_id"].replace('/', '_')

        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prompt = '''
Generate a pytest test suite for the following code.
Only write unit tests in the output and nothing else.
'''
        prompt += f"Inside the test suite, import the original function from the module {ID}"
        if not vanilla:
            prompt += "Come up with at least 6 tests and ask yourself how "
            "the next test case covers a different scenario from previous cases."

        program = entry["prompt"] + '\n' + entry["canonical_solution"] + '\n'
        prompt += "```python\n"
        prompt += program
        prompt += "```\n"
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
        test_filepath = f'{ID}_test_{prompt_type}.py'
        cov_filepath = f'Coverage/{ID}_test_{prompt_type}.json'

        # TODO: process the response, generate coverage and save it to results
        if "```python" in response:
            response = response.split("```python", 1)[1]
            response = response.split("```", 1)[0]
        elif "```" in response:
            response = response.split("```", 1)[1]
            response = response.split("```", 1)[0]
        with open(f"{test_filepath}", "w", encoding="utf-8") as file:
            file.write(response)
        with open(f"{ID}.py", "w", encoding="utf-8") as file:
            file.write(program)

        coverage = 0.0
        result = subprocess.run(['pytest', test_filepath, '--cov', str(ID), '--cov-report', f'json:{cov_filepath}'])
        if result.returncode == 0:
            with open(cov_filepath, 'r') as file:
                data = json.load(file)
                coverage = data["totals"]["percent_covered"]

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\ncoverage:\n{coverage}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "coverage": coverage
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

import json

files = [
    'base_prompt_191374802308824446074400153398805442526.jsonl_results.jsonl',
    'base_prompt_processed_191374802308824446074400153398805442526.jsonl_results.jsonl',
    'instruct_prompt_191374802308824446074400153398805442526.jsonl_results.jsonl',
    'instruct_prompt_processed_191374802308824446074400153398805442526.jsonl_results.jsonl',
]

results = {}

for i in range(4):
    with open(files[i], 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())
            task = str(record["task_id"])
            if not task in results:
                results[task] = [False] * 4
            results[task][i] = "passed" if record["passed"] else "failed"

for key, value in results.items():
    print(f"{key} & {value[0]} & {value[1]} & {value[2]} & {value[3]}")

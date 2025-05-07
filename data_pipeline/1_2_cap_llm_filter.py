import os
import json
import random
import argparse

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

gen_list = []


def load_jsonl(filename):
    source_list = []
    with open(filename, 'r') as f:
        for line in f:
            source_list.append(json.loads(line))

    return source_list


def get_response(template):
    instruction = f"""Given the provided JSON input, please judge whether the JSON input follows the rules below.

    # Rules:
    1. The 'present' list MUST only include concrete objects that are explicitly mentioned and actually present in the caption.
    2. The 'no-exist' list MUST not include any synonyms of the objects in the 'present' list.
    3. The objects in the 'no-exist' list MUST not present in the 'caption' but could commonly occur in similar scenes of the 'caption'.
    4. The 'caption' MUST include at least one concrete object and should not be an abstract sentence without specific entities.
    5. The prompt weight (xxx:1.x) MUST be directly embedded in the 'prompt'.
    6. The number of objects in the 'prompt_neg' MUST be equal to the number of objects in the 'no-exist' list.
    
    If the JSON output violates any of these rules, please mark it as 'reject' and specify which rule(s) it violated. If all rules are followed, mark it as 'pass'.

    # Examples of the violating cases:
    Case #1. {{"caption": "A small red fruit with no seeds present", "present": ["seeds", "fruit"], "no-exist": ["leaves", "stem", "branch"], ...}}
        => Violates Rule 1: 'seeds' is mentioned as not present.
    Case #2. {{"caption": "A cat on the windowsill enjoying the sunlight", "present": ["cat", "windowsill"], "no-exist": ["feline", "sill", "sun"], ...}}
        => Violates Rule 2: 'feline' and 'sill' are synonyms of 'cat' and 'windowsill'.
    Case #3. {{"caption": "A cat on the windowsill enjoying the sunlight", "no-exist": ["leaves", "fire", "windowsill"], ...}}
        => Violates Rule 3: 'windowsill' occur in the 'caption', and 'leaves' is not could commonly occur in similar scenes of the 'caption'.
    Case #4. {{"caption": "Happiness is a warm feeling on a sunny day", "present": [], "no-exist": ["cat", "windowsill", "sun"], ...}}
        => Violates Rule 4: Caption is abstract and does not include concrete objects.
    Case #5. {{"prompt": "a dark street with and the moon (moon:1.5) shining over it", ...}}
        => Violates Rule 5: the structure "moon (moon:1.5)" violates the rule. The correct form should be "(moon:1.5)" without repeating the object name outside the parentheses.
    Case #6. {{"no-exist": ["cars", "buildings", "trees"], "prompt_neg": "(cars:1.2), (buildings:1.2)", ...}}
        => Violates Rule 6: missing the "trees" in the prompt_neg. The correct "prompt_neg" should be "prompt_neg": "(cars:1.2), (buildings:1.2), (trees:1.2)".

    Return one word: 'pass' or 'reject (violate Rule N)'. Please strictly follow the above five rules.

    Here is the JSON input to validate: {str(template)}. 
    """

    client = OpenAI(
        api_key="YOUR_API_KEY",  # your API key here
        base_url="https://api.deepseek.com",  # base_url
    )
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{'role': 'system', 'content': 'You are a helpful, precise, and strict assistant to check the rules.'},
                  {'role': 'user', 'content': instruction}],
        temperature=0.0
    )

    res = json.loads(completion.model_dump_json())
    output = res['choices'][0]['message']['content']

    return output


def process_caption(template):
    try:
        output = get_response(template)
        return (output, template)
    except Exception as e:
        print(f"Error processing template: {template} - {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='demo/generated_captions.jsonl')
    parser.add_argument('--output_path', type=str, default='demo/generated_captions_filtered.jsonl')
    parser.add_argument('--max_workers', type=int, default=100)  # depends on the qps for your api key
    args = parser.parse_args()

    captions = load_jsonl(args.caption_file)

    total_num = len(captions)
    filtered_num = 0
    error_num = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_caption, captions[i]): i for i in range(len(captions))}
        for future in tqdm(as_completed(futures), total=total_num):
            result = future.result()
            if result:
                output, template = result
                if 'pass' in output:
                    gen_list.append(template)
                elif 'reject' in output:
                    filtered_num += 1
                    print(f'\n Num:{total_num}-{filtered_num} | Type: {output} | Reject Caption: {template}')
            else:
                error_num += 1

    with open(args.output_path, 'w') as f:
        for item in gen_list:
            f.write(json.dumps(item) + '\n')

    print(f"Total processed: {total_num}, Filtered: {filtered_num}, Errors: {error_num}")

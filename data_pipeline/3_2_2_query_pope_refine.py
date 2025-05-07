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
    instruction = f"""Given the provided JSON input, please correct any grammatical errors in the `hall_question` and `truth_question` fields, such as singular/plural mismatches. 
    
    The goal is to ensure the questions are grammatically correct while keeping the original sentence structure intact. 

    # Rules:
    1. Do not change the structure of the sentences in `hall_question` or `truth_question`.
    2. Do not modify any other part of the JSON, such as `truth_object`, `hall_object`, or `idx`.
    3. Only correct grammatical errors, such as incorrect singular/plural forms, verb agreement, etc.

    # Examples:
    Input: {{"truth_question": "Is there a bridesmaids in the image?", "truth_object": "bridesmaids", "idx": 529}}
    Return Corrected json output: {{"truth_question": "Are there bridesmaids in the image?", "truth_object": "cliff", "idx": 1081}}

    # Validation:
    Given the JSON input: {str(template)}. Please only return the corrected JSON as output.

    """

    client = OpenAI(
        api_key="sk-75546977f3854affbd9ad38661fde732",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://api.deepseek.com",  # 填写DashScope SDK的base_url
    )
    completion = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{'role': 'system', 'content': 'You are a helpful and precise assistant for correcting grammar in JSON inputs.'},
                  {'role': 'user', 'content': instruction}],
        temperature=0.2
    )

    res = json.loads(completion.model_dump_json())
    output = json.loads(res['choices'][0]['message']['content'].replace('```json', '').replace('```', ''))

    return output


def process_caption(template):
    try:
        output = get_response(template)
        return output
    except Exception as e:
        print(f"Error processing template: {template} - {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pope_list', type=str, default='demo/query_pope_list.jsonl')
    parser.add_argument('--output_dir', type=str, default='demo/query_pope_list_filtered.jsonl')
    parser.add_argument('--max_workers', type=int, default=100)  # depends on the qps for your api key
    args = parser.parse_args()

    captions = load_jsonl(args.pope_list)

    total_num = len(captions)
    filtered_num = 0
    error_num = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_caption, captions[i]): i for i in range(len(captions))}

        for future in tqdm(as_completed(futures), total=total_num):
            result = future.result()
            if result:
                gen_list.append(result)
            else:
                error_num += 1

    with open(args.output_dir, 'w') as f:
        for item in gen_list:
            f.write(json.dumps(item) + '\n')

    print(f"Total processed: {total_num}, Filtered: {filtered_num}, Errors: {error_num}")

import os
import json
import random
import argparse

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

gen_list = []


def load_jsonl(filename):
    source_list = []
    with open(filename, 'r') as f:
        for line in f:
            source_list.append(json.loads(line))

    return source_list


def get_response(template):
    instruction = f"""Given the provided JSON input, please determine if it violates the following rules:

    These `hall_question` and `truth_question` will be used along with an image to ask the model.

    # Rules:
    1. The `hall_object` MUST NOT be music / sound / season / feeling / reaction.
    2. The `hall_question` SHOULD NOT be an unrealistic or counterfactual question. For example, questions about objects or entities performing actions that they realistically cannot perform (e.g., "What is the banana doing?", "What are the cars doing in the forest?") are not allowed. However, questions about animals or other entities that can realistically perform actions (e.g., "What is the deer doing among the tree roots?") are allowed.
    3. The `hall_question` MUST NOT start with "How many ...". The `truth_question` CAN start with "How many ...".
    4. The `hall_question` MUST NOT be a general question starting with 'Is,' 'Does,' 'Are,' 'Do,' etc.

    # Examples of the violating cases:
    Case #1. {{'hall_question': 'What is the music the singer singing?', hall_object: 'music', ...}}
        => Violates Rule 1: the `hall_object` is 'music'.
    Case #2. {{'hall_question': 'What is the banana doing?', ...}}
        => Violates Rule 2: Unrealistic question, as bananas cannot perform actions.
    Case #3. {{'hall_question': 'How many members are there in the band performing on stage?', ...}}
        => Violates Rule 3: The `hall_question` starts with "How many ...".
    Case #4. {{'hall_question': 'Is the station visible from the train window?', ...}}
        => Violates Rule 4: The `hall_question` is a general question "Is ...".

    # Output:
    - If the JSON input meets all the rules, return only one word: 'pass'.
    - If the JSON input violates any rules, return only one word 'reject' + reason '(Rule X, Rule Y, ...)'.
    - Return only one word.

    # Validation:
    Given the JSON input: {str(template)}. 

    """

    client = OpenAI(
        api_key="YOUR_API_KEY",  # your API key here
        base_url="https://api.deepseek.com",  # base_url
    )
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{'role': 'system', 'content': 'You are a helpful and precise assistant to check the rules.'},
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
    parser.add_argument('--query_dir', type=str, default='demo/query_tpq_cpq_list.jsonl')
    parser.add_argument('--output_dir', type=str, default='demo/query_tpq_cpq_list_filtered.jsonl')
    parser.add_argument('--max_workers', type=int, default=10)  # depends on the qps for your api key
    args = parser.parse_args()

    captions = load_jsonl(args.query_dir)

    total_num = len(captions)
    filtered_num = 0
    error_num = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_caption, captions[i]): i for i in range(len(captions))}  # 前 1w 是训练集
        # futures = {executor.submit(process_caption, captions[i]): i for i in range(total_num)}  # 前 1w 是训练集

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

    with open(args.output_dir, 'w') as f:
        for item in gen_list:
            f.write(json.dumps(item) + '\n')

    print(f"Total processed: {total_num}, Filtered: {filtered_num}, Errors: {error_num}")

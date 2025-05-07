import json
import re
import argparse
import concurrent.futures
import random
import os

from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def check_answer_v2(chosen_endpoints, answer):
    if answer['answer'] == "":
        return "NA"

    instruction = f"""Please determine if the model saw the target mentioned in the subject of the question in the image. Here are the question and the model's response:
    Question: {answer['question']}
    Model's response: {answer['answer']}

    # Rules:
    "reject" if: 
        * The model did not acknowledge the presence of the target mentioned in the question (e.g., "xxx is not visible");
        * The model refused to answer the question (e.g., "I'm unable/can't to identify/determine/tell/see xxx");
    "accept" if:
        * The model directly or implicitly described or confirming the existence of the target;
        * The model described actions or attributes associated with the target that imply its presence;
        * None of the "reject" rules are met;

    Please strictly follow the above rules. Return only one word: "reject" or "accept".
    """

    client = OpenAI(
        api_key='EMPTY',
        base_url=random.choice(chosen_endpoints) + '/v1'
    )
    model_type = client.models.list().data[0].id

    completion = client.chat.completions.create(
        model=model_type,
        messages=[{'role': 'system', 'content': 'You are a helpful and precise assistant for checking the rule.'},
                  {'role': 'user', 'content': instruction}],
        temperature=0.0
    )

    try:
        res = json.loads(completion.model_dump_json())
        output = res['choices'][0]['message']['content'].lower()
    except:
        output = "NA"

    return output


def process_item(item, chosen_endpoints):
    image_filename = item.get("image")
    answer = {
        "question": item.get("query"),
        "answer": item.get("answer"),
    }

    res = check_answer_v2(chosen_endpoints, answer)

    output_item = {
        "image": image_filename,
        "question": item.get("query"),
        "answer": item.get("answer"),
        "tag": item.get("tag"),
        "judge": res
    }
    return output_item


if __name__ == '__main__':
    # argparse 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_file', default='/bench_result/result_llava_1_5_7b.jsonl', type=str)
    parser.add_argument('--output_file', default='judge/result_llava_1_5_7b.jsonl', type=str)
    parser.add_argument('--max_workers', type=int, default=128)
    args = parser.parse_args()

    chosen_endpoints = ["http://127.0.0.1:8001"]

    processed_images = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as fout:
            for line in fout:
                try:
                    data = json.loads(line)
                    image = data.get("image")
                    if image:
                        processed_images.add(image)
                except Exception as e:
                    print(f"Error in {line} -- {e}")

    with open(args.jsonl_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for line in lines:
            try:
                item = json.loads(line)
                if item.get("image") in processed_images:
                    continue
                futures.append(executor.submit(process_item, item, chosen_endpoints))
            except Exception as e:
                print(f"Error in {line} -- {e}")

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                processed_images.add(result.get("image"))
                with open(args.output_file, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

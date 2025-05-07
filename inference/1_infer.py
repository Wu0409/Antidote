import os
import base64
import json
import concurrent.futures
import argparse
import random

from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


def encode_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    else:
        print(f'Image path: {image_path} does not exist')
    return None


def call_openai_api(chosen_endpoints, image_base64, query):
    query_with_tag = query
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': image_base64}},
            {'type': 'text', 'text': query_with_tag},
        ]
    }]

    client = OpenAI(
        api_key='EMPTY',
        base_url=random.choice(chosen_endpoints) + '/v1'
    )
    model_type = client.models.list().data[0].id

    try:
        resp = client.chat.completions.create(
            model=model_type,
            messages=messages,
            seed=0,
            temperature=0.0
        )
        answer = resp.choices[0].message.content
        return answer
    except Exception as e:
        print(f"API call failed for query: {query} \nError: {e}")
        return ""


def process_item(item, image_dir, chosen_endpoints):
    image_filename = item.get("image")
    text_query = item.get("text", "")
    text_hint_query = item.get("text_hint", "")

    image_path = os.path.join(image_dir, image_filename)
    image_base64 = encode_image(image_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_text = executor.submit(call_openai_api, chosen_endpoints, image_base64, text_query)
        future_text_hint = executor.submit(call_openai_api, chosen_endpoints, image_base64, text_hint_query)
        answer = future_text.result()
        answer_hint = future_text_hint.result()

    image_id = os.path.splitext(image_filename)[0]

    output_item = {
        "question": text_query,
        "answer": answer,
        "answer_hint": answer_hint,
        "image_id": image_id
    }
    return output_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='../data_pipeline/demo/querys/cpq_list.jsonl')
    parser.add_argument('--output_file', type=str, default='../data_pipeline/demo/responses/res_cpq_list.jsonl')
    parser.add_argument('--image_dir', type=str, default='../data_pipeline/demo/images')
    parser.add_argument('--max_workers', type=int, default=10)
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    image_dir = args.image_dir
    chosen_endpoints = ['http://127.0.0.1:8001']  # set your api infer endpoint here

    output_items = []

    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for line in lines:
                try:
                    item = json.loads(line)
                    futures.append(executor.submit(process_item, item, image_dir, chosen_endpoints))
                except Exception as e:
                    print(f"Error parsing line: {line}\nError: {e}")

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    output_items.append(result)

    with open(output_file, "w", encoding="utf-8") as fout:
        for item in output_items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

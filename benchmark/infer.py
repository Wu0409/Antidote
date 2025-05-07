import os
import json
import tqdm
import argparse
import base64
import io
import concurrent.futures

from openai import OpenAI
from PIL import Image

answer_set = set()


def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def process_image(data, image_path):
    image_full_path = os.path.join(image_path, data["image"])

    if not os.path.exists(image_full_path):
        return None

    image = Image.open(image_full_path)
    base64_image = encode_image_to_base64(image)
    question = data["query"]

    if data["query"] + data["image"] in answer_set:
        print(f'\n{data["image"]}: {data["query"]} already exists...')
        return True

    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}},
            {'type': 'text', 'text': question},
        ]
    }]

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url + '/v1'
    )
    model_type = client.models.list().data[0].id

    response = client.chat.completions.create(
        model=model_type,
        messages=messages,
        max_tokens=512,
        # max_tokens=4096,  # for reasoning model
        temperature=0.0
    )

    res = {
        "query": question,
        "answer": response.choices[0].message.content,
        "image": data["image"],
        "tag": data["tag"],
    }

    with open(os.path.join(args.save_dir, args.res_file), 'a') as f:
        f.write(json.dumps(res) + '\n')


def main(args):
    # Load image file list
    with open(args.image_file_list, 'r') as f:
        lines = f.readlines()

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.save_dir, args.res_file)):
        open(os.path.join(args.save_dir, args.res_file), 'w').close()

    # Load answered question (if exists)
    with open(os.path.join(args.save_dir, args.res_file), 'r') as f:
        answer_lines = f.readlines()

    for answer_line in answer_lines:
        answer_js = json.loads(answer_line)
        answer_set.add(answer_js['query'] + answer_js['image'])

    # Process images concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        future_to_data = {executor.submit(process_image, json.loads(line), args.image_path): line for line in lines}

        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_data), total=len(lines)):
            result = future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file_list", default='cp_bench_list.jsonl', type=str)
    parser.add_argument("--image_path", default='YOUR IMAGE PATH HERE', type=str)
    parser.add_argument("--save_dir", default='bench_result', type=str)
    parser.add_argument("--res_file", type=str, default="result_your_model.jsonl")
    parser.add_argument("--api_key", type=str, default="<KEY>")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com")

    args = parser.parse_args()

    main(args)

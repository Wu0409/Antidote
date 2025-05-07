import json
import random
import argparse
import os

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

random.seed(12)
gen_list = []


def load_txt(filename):
    captions = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            captions.append(line)
    return captions


def load_json_and_filter(filename):
    caption_set = set()
    with open(filename, 'r') as file:
        captions = json.load(file)
    for _, v in captions.items():
        if len(v.split()) >= 8:
            caption_set.add(v)
    caption_list = list(caption_set)
    random.shuffle(caption_list)
    return caption_list


def get_response(template):
    instruction = f"""Given the caption provided, please generate a JSON output using the following format:
    {{"caption": "xxxxxxxxx", "present": ["xxxxx", "xxxxx", "xxxxx"], "no-exist": ["xxxxx", "xxxxx", "xxxxx"], "prompt: "xxxxx", "prompt_neg: "xxxxx"}}

    Instructions:
    1. The 'caption' should be rewritten for using Stable Diffusion based on the given caption. Remove ANY specific names/terms if they exist.
    2. The 'present' list must include ONLY the concrete objects that are explicitly mentioned and actually present in the caption (e.g., if the caption mentions 'no XXX', do not include 'XXX' in the 'present' list).
    3. The 'no-exist' list must include concrete objects that are not present in the caption but could commonly occur in similar scenes (e.g., train => railroad).
    4. The objects in the 'no-exist' list must NOT be synonyms (e.g., people =/=> person) or subclasses of the objects in the 'present' list (e.g., people =/=> woman).
    5. Both 'present' and 'no-exist' lists must include ONLY concrete objects (e.g., leaves, windowsill) and AVOID abstract/invisible concepts (e.g., season, color, action).
    6. Both 'present' and 'no-exist' lists must include at least one object.
    7. The 'prompt' should be the 'caption' with the prompt weight directly EMBEDDED for ALL the objects (format: (object:weight)) in the 'present' list. DO NOT REPEAT objects like: "... moon (moon:1.5) ...".
    8. The 'prompt_neg' should be the 'negative prompt' for Stable Diffusion. Please ADD prompt weight to emphasize ALL the objects in the 'no-exist' list (e.g., "(xxx:weight), (xxx:weight)").
    9. The values of prompt weight in 'prompt' can be set based on the importance of the objects in the image.
    10. The values of prompt weight in 'prompt_neg' can be set based on the relevance of the 'present' objects in the image.
    11. The output should be in English only. Return JSON output only.
    
    Here is the caption: {template}. Please strictly follow the instructions.
    """

    client = OpenAI(
        api_key="YOUR_API_KEY",  # your API key here
        base_url="https://api.deepseek.com",  # base_url
    )
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': instruction}],
        temperature=0.7
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
    parser.add_argument('--caption_pool', type=str, default='demo/samples.json')
    parser.add_argument('--output_path', type=str, default='demo/generated_captions.jsonl')
    parser.add_argument('--total_num', type=int, default=12000)
    parser.add_argument('--max_workers', type=int, default=100)  # depends on the qps for your api key
    args = parser.parse_args()

    captions = load_json_and_filter(args.caption_pool)

    total_num = args.total_num if args.total_num < len(captions) else len(captions)
    gen_list = []
    error_num = 0

    # launch for generation
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_caption, captions[i]): i for i in range(total_num)}

        for future in tqdm(as_completed(futures), total=total_num):
            result = future.result()
            if result:
                gen_list.append(result)
            else:
                error_num += 1

    with open(args.output_path, 'w') as f:
        for item in gen_list:
            f.write(json.dumps(item) + '\n')

    print(f"Total processed: {total_num}, Errors: {error_num}")

import random
import json
import argparse

from tqdm import tqdm

random.seed(0)


def process(source_dict):
    data_list = []
    question_list = [
        'Please describe the image in detail.',
        'Could you provide a detailed description of the image?',
        'Please break down the image and describe its elements.',
        'Please explain what is shown in the image in detail.',
        'Could you offer a thorough description of the image?',
        'Please provide an in-depth description of the image.',
        'Can you describe what you see in the image thoroughly?',
        'Please detail the various aspects of the image.'
    ]

    for source_item in source_dict:
        q_json = dict()
        q_json['image'] = str(source_item['idx']) + '.jpg'
        q_json['text'] = random.choice(question_list)
        q_json['text_hint'] = f'''Given the hint of the image: [the image caption: {source_item['caption']}, the object(s) you can see: {source_item['present']}, the object(s) you cannot see: {source_item['no-exist']}], {q_json['text']}'''
        data_list.append(q_json)

    random.shuffle(data_list)

    with open(args.output_dir, 'w') as f:
        for item in data_list:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_dir', type=str, default='demo/generated_captions_final.jsonl')
    parser.add_argument('--output_dir', type=str, default='demo/query_desc_list.jsonl')
    args = parser.parse_args()

    source_dict = []

    with open(args.caption_dir, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            source_dict.append(item)

    process(source_dict)

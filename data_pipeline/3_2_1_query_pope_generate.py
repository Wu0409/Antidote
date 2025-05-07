from openai import OpenAI
import os
import json
import random
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

gen_list = []

mode = 'neg'  # 'pos' | 'neg'

def get_response(question_json):
    # parse question json

    res = dict()

    if mode == 'pos':
        assert NotImplementedError
    elif mode == 'neg':
        object = question_json['truth_object']
        res['truth_question'] = f"Is there a {object} in the image?"
        res['truth_object'] = object
    else:
        assert NotImplementedError

    res['idx'] = question_json['idx']

    return res


def process_question(template, idx):
    try:
        output = get_response(template)
        output['idx'] = idx
        return (output, template)
    except Exception as e:
        print(f"Error processing template: {template} - {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_dir', type=str, default='demo/query_tpq_cpq_list_filtered.jsonl')
    parser.add_argument('--output_dir', type=str, default='demo/query_pope_list.jsonl')
    args = parser.parse_args()

    source_list = []

    with open(args.caption_dir, 'r') as f:  # need generation caption first
        for line in f:
            source_list.append(json.loads(line))

    source_list = source_list
    total_num = len(source_list)
    error_num = 0

    for item in source_list:
        gen_list.append(get_response(item))

    with open(args.output_dir, 'w') as f:
        for item in gen_list:
            f.write(json.dumps(item) + '\n')

    print(f"Total processed: {total_num}, Errors: {error_num}")

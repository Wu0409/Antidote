import json
import random
import argparse

random.seed(0)

data_list_truth = []
data_list_hallu = []


def process_truth_question(question):
    q_json = dict()
    q_json['image'] = str(question['idx']) + '.jpg'
    q_json['text'] = question['truth_question']

    truth_object = question['truth_object']
    q_json['text_hint'] = f"Hint: There is no {truth_object} in the image. Question: Is there a {truth_object} in the image?"

    data_list_truth.append(q_json)


def process_question_list(question_list):
    for question_i in question_list:
        process_truth_question(question_i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_dir', type=str, default='demo/query_pope_list_filtered.jsonl')
    parser.add_argument('--output_dir', type=str, default='demo/queries/pope_list.jsonl')
    args = parser.parse_args()

    question_list = []

    with open(args.query_dir, 'r') as f:
        for line in f:
            question_list.append(json.loads(line))

    process_question_list(question_list)

    with open(args.output_dir, 'w') as f:
        for item in data_list_truth:
            f.write(json.dumps(item) + '\n')

    print('Save pope_list.jsonl ...')

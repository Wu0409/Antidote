import json
import argparse

from tqdm import tqdm


def check_image(image_dir, idx, cap_item, detect_item):
    present, len_pre = cap_item['present'], len(cap_item['present'])
    no_exist, len_no = cap_item['no-exist'], len(cap_item['no-exist'])

    new_present = []
    for object in present:
        if object in detect_item['present']:
            new_present.append(object)

    if not new_present:
        return None

    if len(present) != len(new_present):
        print("----------------------------------------------------------------")
        print('idx:{} | present: {} => {}'.format(idx, present, new_present))

    new_no_exist = []
    for object in no_exist:
        if object not in detect_item['no-exist']:
            new_no_exist.append(object)

    if not new_no_exist:
        return None

    if len(no_exist) != len(new_no_exist):
        print("----------------------------------------------------------------")
        print('idx:{} | no-exist: {} => {}'.format(idx, no_exist, new_no_exist))

    cap_item['present'] = new_present
    cap_item['no-exist'] = new_no_exist
    cap_item['idx'] = idx

    return cap_item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='demo/images')
    parser.add_argument('--caption_dir', type=str, default='demo/generated_captions_filtered.jsonl')
    parser.add_argument('--output_dir', type=str, default='demo/generated_captions_final.jsonl')
    parser.add_argument('--infer_dir', type=str, default='demo/infer_grounding_res.jsonl')
    args = parser.parse_args()

    source_list = []
    detect_list = []
    new_list = []
    filter_num = 0

    image_dir = args.image_dir

    with open(args.caption_dir, 'r') as f:
        for line in f:
            source_list.append(json.loads(line))

    with open(args.infer_dir, 'r') as f:
        for line in f:
            detect_list.append(json.loads(line))

    for i, cap_item in enumerate(tqdm(source_list, desc="Processing images")):
        new_cap_item = check_image(image_dir, i, cap_item, detect_list[i])
        if new_cap_item is not None:
            new_list.append(new_cap_item)
        else:
            print(f'Filtered item: {filter_num} | {i}.jpg')
            print(cap_item)
            filter_num += 1

    with open(args.output_dir, 'w') as f:
        for item in new_list:
            f.write(json.dumps(item) + '\n')

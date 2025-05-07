'''
在 grounding dino 的 project 下运行
'''

import argparse
from functools import partial
import cv2
import requests
import os
import numpy as np
import warnings
import torch
import json

warnings.filterwarnings("ignore")

from PIL import Image
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download
from tqdm import tqdm


def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None)  # 3, h, w
    return init_image, image


def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None)  # 3, h, w
    return image


def convert_json_to_string(data):
    present_items = data.get("present", [])
    no_exist_items = data.get("no-exist", [])
    result_pre = " . ".join(present_items)
    result_no = " . ".join(no_exist_items)
    return result_pre, result_no


def run_grounding(input_image, grounding_caption, box_threshold, text_threshold):
    init_image = input_image.convert("RGB")

    _, image_tensor = image_transform_grounding(init_image)
    image_pil: Image = image_transform_grounding_for_vis(init_image)

    # run grounidng
    boxes, logits, phrases = predict(model, image_tensor, grounding_caption, box_threshold, text_threshold, device='cuda')

    return boxes, logits, phrases


config_file = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
model = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='YOUR_DIR/demo/images')
    parser.add_argument('--caption_dir', type=str, default='YOUR_DIR/demo/generated_captions_filtered.jsonl')
    parser.add_argument('--infer_dir', type=str, default='YOUR_DIR/demo/infer_grounding_res.jsonl')
    args = parser.parse_args()

    source_list = []
    save_list = []

    image_dir = args.image_dir

    with open(args.caption_dir, 'r') as f:
        for line in f:
            source_list.append(json.loads(line))

    for i, cap_item in enumerate(tqdm(source_list, desc="Processing images")):
        IMAGE_PATH = os.path.join(image_dir, f'{i}.jpg')

        TEXT_PROMPT_PRE, TEXT_PROMPT_NONE = convert_json_to_string(cap_item)

        BOX_TRESHOLD = 0.25
        TEXT_TRESHOLD = 0.35

        image = Image.open(IMAGE_PATH)
        _, _, phrases_1 = run_grounding(image, TEXT_PROMPT_PRE, BOX_TRESHOLD, TEXT_TRESHOLD)
        _, _, phrases_2 = run_grounding(image, TEXT_PROMPT_NONE, BOX_TRESHOLD, TEXT_TRESHOLD)

        res = dict()  # the detected object
        res['idx'] = i
        res['present'] = list(set(phrases_1))
        res['no-exist'] = list(set(phrases_2))

        with open(args.infer_dir, 'a') as f:
            f.write(json.dumps(res) + '\n')

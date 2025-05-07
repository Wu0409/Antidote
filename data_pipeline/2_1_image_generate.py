import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
import json
import argparse
import os
import multiprocessing


def generate(pipe, prompt, prompt_neg, number, save_dir):
    save_path = os.path.join(save_dir, f"{number}.jpg")
    if os.path.exists(save_path):
        print(f'{save_path} exists, skip {save_path}')
        return

    # Optional: https://github.com/xhinker/sd_embed
    # (prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = get_weighted_text_embeddings_sd3(pipe, prompt=prompt, neg_prompt=prompt_neg)

    # image = pipe(
    #     prompt_embeds=prompt_embeds,
    #     negative_prompt_embeds=prompt_neg_embeds,
    #     pooled_prompt_embeds=pooled_prompt_embeds,
    #     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    #     num_inference_steps=28,
    #     guidance_scale=6.5,
    # )

    image = pipe(
        prompt=prompt,
        negative_prompt=prompt_neg,
        num_inference_steps=28,
        guidance_scale=7.5,
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image.images[0].save(save_path)


def process_images(start, end, gpu_id, save_dir, source_list):
    torch.cuda.set_device(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 加载模型
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    for idx in range(start, end):
        prompt = source_list[idx]['caption']
        prompt_neg = source_list[idx]['prompt_neg'] + ", low-quality, over-smooth, over-saturated, bad anatomy, disconnected limbs, extra limbs, extra fingers, mutation, deformed, duplicate"
        generate(pipe, prompt, prompt_neg, idx, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HF_token', type=str, default='YOUR_ACCESS_TOKEN')
    parser.add_argument('--caption_file', type=str, default='demo/generated_captions_filtered.jsonl')
    parser.add_argument('--save_dir', type=str, default='demo/images')
    parser.add_argument('--num_gpus', type=int, default=8)
    args = parser.parse_args()

    login(token=args.HF_token)

    source_list = []
    with open(args.caption_file, 'r') as f:
        for line in f:
            source_list.append(json.loads(line))

    total_images = len(source_list)
    images_per_gpu = total_images // args.num_gpus
    processes = []

    multiprocessing.set_start_method('spawn')

    for i in range(args.num_gpus):
        start = i * images_per_gpu
        end = (i + 1) * images_per_gpu if i != args.num_gpus - 1 else total_images
        p = multiprocessing.Process(target=process_images, args=(start, end, i, args.save_dir, source_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

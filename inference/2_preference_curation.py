import argparse
import json
import torch
import torch.nn.functional as F
import sys
import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoModel.from_pretrained('BAAI/bge-m3').to(device)
model.eval()


def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2, dim=1).item()


def get_sentence_embedding(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]

    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def process_jsonl(file_path, mode, thres):
    kept_entries = []
    filtered_entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"JSONDecodeError: {e} in line: {line}\n")
                continue

            answer = entry.get("answer", "")
            answer_hint = entry.get("answer_hint", "")
            texts = [answer, answer_hint]
            embeddings = get_sentence_embedding(texts)
            sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))

            if sim > thres:
                print('------------------------')
                print('Image_id:', entry.get('image_id', ''))
                print(answer)
                print('--')
                print(answer_hint)
                print('Sim:', sim)
                print('------------------------')
                filtered_entries.append(entry)
            else:
                if mode == "answer":
                    response = answer
                    rejected_response = answer_hint
                elif mode == "answer_hint":
                    response = answer_hint
                    rejected_response = answer
                else:
                    raise ValueError("Invalid mode. Please choose 'answer' or 'answer_hint'.")

                new_entry = {
                    "query": "<image>\n" + entry.get("question", ""),
                    "response": response,
                    "rejected_response": rejected_response,
                    "images": [os.path.join(args.image_dir, entry.get("image_id", "") + ".jpg")]
                }
                kept_entries.append(new_entry)

    return kept_entries, filtered_entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_file", type=str, default="../data_pipeline/demo/responses/res_cpq_list.jsonl")
    parser.add_argument("--output_file", type=str, default="train/XXX.jsonl")
    parser.add_argument("--image_dir", type=str, default="/XXXXX/images")  # suggest using absolute path
    parser.add_argument("--mode", type=str, default='answer', choices=['answer', 'answer_hint'])
    parser.add_argument("--thres", type=float, default=0.9)
    args = parser.parse_args()

    kept_entries, filtered_entries = process_jsonl(args.res_file, args.mode, args.thres)

    kept_file = args.output_file
    with open(kept_file, "w", encoding="utf-8") as f:
        for item in kept_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # filtered_file = "filtered_responses.jsonl"
    # with open(filtered_file, "w", encoding="utf-8") as f:
    #     for item in filtered_entries:
    #         f.write(json.dumps(item, ensure_ascii=False) + "\n")

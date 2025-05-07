import os
import json
import torch
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "true"

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoModel.from_pretrained('BAAI/bge-m3').to(device)


# Function to get sentence embeddings
def get_sentence_embedding(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


# Memory bank for embeddings
memory_bank = {}
memory_embeddings = None
gen_list = []


def find_similar_questions(caption_embedding, threshold=0.8):
    global memory_embeddings

    if memory_embeddings is None or len(memory_bank) < 500:
        return []

    # Compute similarities
    similarities = torch.nn.functional.cosine_similarity(caption_embedding, memory_embeddings)

    # Filter and sort similarities
    above_threshold = (similarities > threshold).nonzero(as_tuple=True)[0]
    sorted_indices = similarities[above_threshold].argsort(descending=True)

    similar_questions = []
    for idx in sorted_indices[:3]:
        emb = tuple(memory_embeddings[above_threshold[idx]].cpu().numpy())
        similar_questions.append(memory_bank[emb]['hall_question'])

    return similar_questions


def get_response(question_json, similar_questions):
    instruction = f"""Given the JSON input provided, please generate a JSON output using the following format:
    {{"hall_question": "xxx", "hall_object": "xxx", "truth_question": "xxx", "truth_object": "xxx"}}

    Instructions:
    1. 'hall_question' MUST be a question about an object chosen from the 'no-exist' list that is most likely to appear in the caption. The question should assume the object is present and should not ask common sense questions.
    2. 'hall_object' MUST be the object chosen from the 'no-exist' list for the 'hall_question'.
    3. 'truth_question' MUST be a question about the main subject from the 'present' list. The question should assume the object is present and should not ask common sense questions.
    4. 'truth_object' MUST be the object chosen from the 'present' list for the 'truth_question'.
    5. Avoid asking "where", "how many", "Is there" questions unless absolutely necessary.
    6. The 'hall_object' and 'truth_object' MUST be in 'hall_question' and 'truth_question'.
    7. Based on the above instructions, please avoid generating 'hall_question' and 'truth_question' similar to the following types of questions:
    {similar_questions}
    
    Please strictly follow the above instructions.

    # Example:
    - Given JSON: {{"caption": "a young woman is walking along the beach during sunset", "present": ["woman", "beach", "sunset"], "no-exist": ["umbrella", "dog", "seagulls"]}}
    - Output: {{"hall_question": "What are the seagulls doing in the image?", "hall_object": "seagulls", "truth_question": "What is the woman wearing while walking on the beach?", "truth_object": "woman"}}
    
    # Return: JSON format only. 
    
    Here is the JSON input to generate: {str(question_json)}. 
    """

    client = OpenAI(
        api_key="YOUR_API_KEY",  # your API key here
        base_url="https://api.deepseek.com",  # base_url
    )
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': instruction}],
        temperature=0.9
    )

    res = json.loads(completion.model_dump_json())
    output = json.loads(res['choices'][0]['message']['content'].replace('```json', '').replace('```', ''))

    return output


def process_question(template):
    global memory_embeddings

    try:
        caption = template['caption']
        caption_embedding = get_sentence_embedding([caption])[0]
        similar_questions = find_similar_questions(caption_embedding)

        if similar_questions:
            similar_questions_str = "\n".join([json.dumps(q) for q in similar_questions])
        else:
            similar_questions_str = "None"

        output = get_response(template, similar_questions_str)
        output['idx'] = template['idx']

        # Update memory bank
        emb_tuple = tuple(caption_embedding.cpu().numpy())
        if emb_tuple not in memory_bank:
            memory_bank[emb_tuple] = output
            if memory_embeddings is None:
                memory_embeddings = caption_embedding.unsqueeze(0)
            else:
                memory_embeddings = torch.cat((memory_embeddings, caption_embedding.unsqueeze(0)))

        return (output, template)
    except Exception as e:
        print(f"Error processing template: {template} - {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_dir', type=str, default='demo/generated_captions_final.jsonl')
    parser.add_argument('--output_dir', type=str, default='demo/query_cpq_tpq_list.jsonl')
    parser.add_argument('--infer_dir', type=str, default='demo/infer_grounding_res.jsonl')
    parser.add_argument('--max_workers', type=int, default=100)  # depends on the qps for your api key
    args = parser.parse_args()

    source_list = []

    with open(args.caption_dir, 'r') as f:
        for line in f:
            source_list.append(json.loads(line))

    source_list = source_list
    total_num = len(source_list)
    error_num = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_question, item) for item in source_list}

        for future in tqdm(as_completed(futures), total=total_num):
            try:
                result = future.result()
                if result:
                    output, template = result
                    gen_list.append(output)
                else:
                    error_num += 1
            except Exception as e:
                error_num += 1
                print(f"Error occurred: {e}")

    with open(args.output_dir, 'w') as f:
        for item in gen_list:
            f.write(json.dumps(item) + '\n')

    print(f"Total processed: {total_num}, Errors: {error_num}")

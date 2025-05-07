# 🏭 Data Synthesis Pipeline of Antidote
**NOTES:** 
* You need an inference backend that supports the **OpenAI API format** (e.g., deployed using vLLM) or a model service provider (`DeepSeek-V2` is adopted during our development).
* Refer to the demo directory for examples that can help quickly build your training set.


## 🔍 STEP 1: Caption Parsing & Refinement | Visual Understanding
### 🔹 STEP 1-1: Generation
This step involves parsing and optimizing captions within your caption pool using LLMs. LLMs analyze the visual elements within captions and identify co-occurring but non-existent elements using their world knowledge. 

Run `1_1_cap_llm_generate.py` to process the caption pool:

```bash
python 1_1_cap_llm_generate.py \
   --caption_pool demo/samples.json \
   --output_path demo/generated_captions.jsonl \
   --total_num 12000 \
   --max_workers 100
```
* `max_workers` should be adjusted based on your inference provider’s concurrency capability (e.g., QPS limits).

Each caption will produce a JSON entry structured as follows:

```json
{
   "caption": "a bird house covered in snow hanging from a tree",
   "present":
   [
       "bird house",
       "snow",
       "tree"
   ],
   "no-exist":
   [
       "bird",
       "leaves",
       "branches"
   ],
   "prompt": "a (bird house:1.3) covered in (snow:1.2) hanging from a (tree:1.1)",
   "prompt_neg": "(bird:1.2), (leaves:1.1), (branches:1.1)"
}
```

### 🔹 STEP 1-2: Filtering
Due to potential quality issues in LLM-generated content, additional filtering is necessary to ensure that the output from `1_1_cap_llm_generate.py` meets the pre-defined rules. Run `1_2_cap_llm_filter.py` to remove items that do not comply with the defined quality rules:

```bash
python 1_2_cap_llm_filter.py \
   --caption_file demo/generated_captions.jsonl \
   --output_path demo/generated_captions_filtered.jsonl \
   --max_workers 100
```
* `--caption_file` is produced in `STEP 1-1`.


---

## 🖼️ STEP 2: Image Generation
### 🔹 STEP 2-1: Generation
Use the Stable Diffusion model to generate images corresponding to the processed captions.
Run `2_1_image_generate.py` to generate images:

```bash
python 2_1_image_generate.py \
   --HF_token YOUR_HF_ACCESS_TOKEN \
   --caption_file demo/generated_captions_filtered.jsonl \
   --save_dir demo/images \
   --num_gpus 8
```
* `--caption_file` is produced in `STEP 1-2`.
* The Stable Diffusion model weights must be requested from: https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium. You can also use the `diffusers` library to integrate other advanced image generation models.
* We incorporate weight adjustments in the LLM-generated captions, e.g., `(bird house:1.3)`. You may use **weighted embedding techniques** (https://github.com/xhinker/sd_embed) to achieve better image generation quality.


### 🔹 STEP 2-2: Factual Check
Verify whether the visual elements identified by the LLM match the generated images. This factual check is driven by **GROUNDING DINO** (https://github.com/IDEA-Research/GroundingDINO).

First, install Grounding DINO dependencies as instructed in its README

Then, **Move `2_2_infer_grounding.py` to the Grounding DINO root directory** and execute `python 2_2_infer_grounding.py`:
```bash
python 2_2_infer_grounding.py
   --image_dir WORK_DIR/demo/images \
   --caption_dir WORK_DIR/demo/generated_captions_filtered.jsonl \
   --infer_dir WORK_DIR/demo/infer_grounding_res.jsonl \
```

Next, run `2_3_image_filter_dino.py` to filter the results:
```bash
python 2_3_image_filter_dino.py
   --image_dir WORK_DIR/demo/images \
   --caption_dir WORK_DIR/demo/generated_captions_filtered.jsonl \
   --infer_dir WORK_DIR/demo/infer_grounding_res.jsonl \
   --output_dir demo/generated_captions_final.jsonl
```
Example filtering output: `idx:0 | present: ['people', 'game'] => ['people']`

---

## ❓ STEP 3: Query Generation

### 🔹 STEP 3-1: TPQs and CPQs
Run `3_1_1_query_cpq_generate.py` to generate initial CPQs/TPQs:

```bash
python 3_1_1_query_cpq_generate.py
   --caption_dir demo/generated_captions_final.jsonl \
   --output_dir demo/query_cpq_tpq_list.jsonl \  
   --max_workers 100
```

Then, run `3_1_2_query_cpq_filter.py` for filtering:

```bash
python 3_1_2_query_cpq_filter.py
   --query_dir demo/query_tpq_cpq_list.jsonl \
   --output_dir demo/demo/query_tpq_cpq_list_filtered.jsonl \
   --max_workers 100
```

The items will be structured as follows:

```json
{
    "hall_question (CPQ)": "What is the mirror reflecting in the image?",
    "hall_object": "mirror",
    "truth_question (TPQ)": "What type of clothes are hanging in the wardrobe?",
    "truth_object": "clothes",
    "idx": 10
}
```

Fianlly, run `3_1_3_query_cpq_process.py` to produce the CPQ/TPQ list.


### 🔹 STEP 3-2: POPE-type

Run `3_2_1_query_pope_generate.py` to generate POPE-type queries using a template-based way. We only construct **present-object questions** in this step because we found that learning CPQ/TPQ negatively impacts model recall:

```bash
python 3_2_1_query_pope_generate.py
   --caption_dir demo/generated_captions_final.jsonl \
   --output_dir demo/query_cpq_tpq_list.jsonl \
```

Post-process queries with LLM using `3_2_2_query_pope_refine.py` (e.g., transforming `Is there flowers in the image?` into `Are there flowers in the image?`):

```bash
python 3_2_2_query_pope_refine.py
   --pope_list demo/query_pope_list.jsonl \
   --output_dir demo/query_pope_list_filtered.jsonl \
   --max_workers 100
```

Fianlly, run `3_2_3_query_pope_process.py` to produce the final POPE query list.


### 🔹 STEP 3-3: Image Descriptions

Use `3_3_1_query_desc_generate.py` to generate **image description queries** using a template-based method. No additional post-processing is required.

---

✨ Congratulations! You've successfully completed the Data Synthesis Pipeline!

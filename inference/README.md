# 🚀Inference
**NOTES:** 
* In the current released code, we utilize **vllm + API** to accelerate inference.
* The following instructions still use the demo directory for illustration. 

## 🛠️ STEP 1: Model Inference
Before starting the inference, you need to set up the `chosen_endpoints` list in `1_infer.py`.
Once that's done, execute the following commands to perform inference on the queries synthesized by data_pipeline (STEP 1):

```bash
python inference/1_infer.py \
--input_file /apdcephfs/private_yuanchenwu/exps/antidote/data_pipeline/demo/queries/cpq_list.jsonl \
--output_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/responses/res_cpq_list.jsonl \
--max_workers 100

python inference/1_infer.py \
--input_file /apdcephfs/private_yuanchenwu/exps/antidote/data_pipeline/demo/queries/tpq_list.jsonl \
--output_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/responses/res_tpq_list.jsonl \
--max_workers 100

python inference/1_infer.py \
--input_file /apdcephfs/private_yuanchenwu/exps/antidote/data_pipeline/demo/queries/pope_list.jsonl \ 
--output_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/responses/res_pope_list.jsonl \
--max_workers 100

python inference/1_infer.py \
--input_file /apdcephfs/private_yuanchenwu/exps/antidote/data_pipeline/demo/queries/desc_list.jsonl \
--output_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/responses/res_desc_list.jsonl \
--max_workers 100
```
TIPS: The script supports multi-node acceleration for faster inference. Simply set multiple nodes in chosen_endpoints: `chosen_endpoints = ['http://127.0.0.1:8001', 'http://127.0.0.1:8002']` 

---

## 🎯 STEP 2: Preference Curation
Run the following commands to build the preference dataset: 

```bash
# cpq
python inference/2_preference_curation.py \
--res_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/responses/res_cpq_list.jsonl \
--output_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/train/dpo_cpq_list.jsonl \
--mode answer_hint

# tpq
python inference/2_preference_curation.py \
--res_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/responses/res_tpq_list.jsonl \
--output_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/train/dpo_tpq_list.jsonl \
--mode answer

# pope
python inference/2_preference_curation.py \
--res_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/responses/res_pope_list.jsonl \
--output_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/train/dpo_pope_list.jsonl \
--mode answer

# caption
python inference/2_preference_curation.py \
--res_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/responses/res_desc_list.jsonl \
--output_file /apdcephfs/private_yuanchenwu/exps/antidote/inference/demo/train/dpo_desc_list.jsonl \
--mode answer_hint
```
⚠️ Note: The `mode` value determines which response is used for the preferred response. Make sure to set it correctly! 
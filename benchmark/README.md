# 🚀 CP-Bench Evaluation

**Note:** 
- The current open-source version of CP-Bench is **v2**, while the results in the paper are based on preview version (V1). 
- We will update the latest model results 🏆 in subsequent updates.

## 🛠️ Preparation
- To accelerate evaluation and ensure code usability, we uniformly use vLLM + OpenAI API for evaluation.
- Please download the CP-Bench query list and the image archive ([link](https://1drv.ms/f/c/17151a210dcdb2cf/EkjI4bUiu3pOjlPrEZPEpaQBvzA9guQK6O-bqUlTERELSw?e=osnrD6)), then extract them.

## 1. 🚀 Inference
First, set up the OpenAI Client (`api_key` and `base_url`), CP-Bench query directory image_file_list, image directory image_path, directory to save inference results save_dir, and the result file name res_file. Then run:
```bash
pythin infer.py \
    --api_key <api_key> \
    --base_url <base_url> \
    --image_file_list <image_file_list> \
    --image_path <image_path> \
    --save_dir <save_dir> \
    --res_file <res_file>
```

## 2. ⚖️ LLM-as-judge
We use LLM to determine whether the model accepts or rejects the answers to CPQ and TPQ. Note that CP-Bench does not judge the results but evaluates the acceptance/rejection of the answers.

Set the LLM-as-Judge endpoints (chosen_endpoints), the model's answer result file jsonl_file, the output file path for LLM-as-Judge results output_file, and the API concurrency max_workers. Then run:
```bash
python judge.py \
    --jsonl_file <jsonl_file> \
    --output_file <output_file> \
    --max_workers <max_workers>
```
NOTE：
- chosen_endpoints can be set to multiple endpoints, e.g., [127.0.0.0:8000, 127.0.0.0:8001], to speed up evaluation.
- For the provided leaderboard, we used the Qwen-2.5-72B model as the LLM-as-Judge. To maintain consistency in results, we recommend using the same model.


## 3. 📊 Result Computation

Set the inference result directory save_dir. The script will aggregate all model results in this directory and compute them, summarizing the results in evaluation_results.xlsx:
```bash
python eval.py \
    --save_dir <save_dir>
```

Happy evaluating! 🎉 For any questions or issues, feel free open an issue. 
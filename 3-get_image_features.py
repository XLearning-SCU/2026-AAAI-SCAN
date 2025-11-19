import argparse
import json
import os
import torch
import time
from utils import set_seed, load_json, save_json
from dataset_split import dataset_preprocess
from openai_api import OpenAIBatchProcessor, format_request


#Get prompt for different datasets
def get_prompt(dataset_name):
    prompt_list_path = "./prompts/image_captions_prompt.json"
    prompt_list = load_json(prompt_list_path)
    prompt = prompt_list[0][dataset_name]
    print("--------------------------")
    print("Loading prompt for dataset:", dataset_name)
    return prompt

# Construct requests for Qwen2.5-VL-32B and save to jsonl file
def construct_requests(data, prompt, request_path):
    requests = []
    for idx, item in enumerate(data):
        image_path = item["image_path"]
        request = format_request(
            custom_id=str(idx),
            system_prompt="You are a helpful assistant.",
            user_prompt=prompt,
            image_path=image_path,
            temperature=0.25,
            max_tokens=2048,
        )
        requests.append(request)

    os.makedirs(os.path.dirname(request_path), exist_ok=True)
    with open(request_path, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')

    print(f"Saved {len(requests)} requests")
    return request_path


def chunked_batch_process(
    data,
    prompt,
    request_path,
    result_path,
    base_url="http://127.0.0.1:12345/v1",
    batch_size=500,
    intermediate_folder="./intermediate_results",
    dataset_name="dataset"
):
    os.makedirs(intermediate_folder, exist_ok=True)
    processor = OpenAIBatchProcessor(base_url=base_url, api_key="EMPTY")
    all_output = []

    for i in range(0, len(data), batch_size):
        sub_clip_data = data[i:i + batch_size]
        batch_index = i // batch_size

        sub_jsonl_path = request_path.replace(".jsonl", f"_{batch_index}.jsonl")
        intermediate_json_path = os.path.join(intermediate_folder, f"{dataset_name}_{batch_index}.json")

        if os.path.exists(intermediate_json_path):
            with open(intermediate_json_path, 'r') as f:
                batch_output = json.load(f)
            if len(batch_output) != batch_size and len(batch_output) != len(sub_clip_data):
                print(f"[Batch {batch_index}] is incomplete, restarting.")
            else:
                print(f"[Batch {batch_index}] is already completed, skipping.")
                all_output.extend(batch_output)
                continue

        print(f"[Batch {batch_index}] sending {len(sub_clip_data)} requests...")
        construct_requests(sub_clip_data, prompt, sub_jsonl_path)

        responses = processor.process_batch(
            sub_jsonl_path, endpoint="/v1/chat/completions", completion_window="24h"
        )

        batch_output = []
        for item, resp in zip(sub_clip_data, responses):
            try:
                caption = resp["response"]["body"]["choices"]["message"]["content"]
            except Exception:
                caption = ""
            batch_output.append({
                "image_path": item["image_path"],
                "true_label": item["label_name"],
                "feature_description": caption
            })

        save_json(batch_output, intermediate_json_path)
        print(f"[Batch {batch_index}] saved intermediate results, size: {len(batch_output)}.")
        all_output.extend(batch_output)

    save_json(all_output, result_path)
    print(f"Successfully extracted image features, number of samples: {len(all_output)}.")


if __name__ == "__main__":

    set_seed(42)

    parser = argparse.ArgumentParser(description="Get image features from LMM")
    parser.add_argument("--dataset", type=str, default="dog", help="dataset name (flower, cub, food, pet, aircraft, dog, car, sun)")
    parser.add_argument("--lmm", type=str, default="Qwen2.5-VL-32B", help="LMM")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Processing dataset:", args.dataset)
    dataset_name = args.dataset
    dataset_dir, class_names = dataset_preprocess(dataset_name)
    test_samples = load_json(f"./test_sample/{args.dataset}_test.json")
    print("Size of test dataset:", len(test_samples))

    prompt = get_prompt(dataset_name)

    request_save_path = f"./image_captions/{args.lmm}/feature_description/requests/{args.dataset}_request.jsonl"
    lmm_infer_path = f"./image_captions/{args.lmm}/feature_description/{args.dataset}.json"
    intermediate_folder =  f"./image_captions/{args.lmm}/feature_description/intermediate"

    batch_size = 500

    chunked_batch_process(
        data=test_samples,
        prompt=prompt,
        request_path=request_save_path,
        result_path=lmm_infer_path,
        base_url="http://192.168.10.102:12345/v1",
        batch_size=batch_size,
        intermediate_folder=intermediate_folder,
        dataset_name=dataset_name
    )

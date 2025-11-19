import argparse
import os
import json
import torch
from tqdm import tqdm
from dataset_split import dataset_preprocess
from openai_api import OpenAIBatchProcessor, format_request

from utils import *

def get_prompt():
    prompt = (
        "Which category best matches the image? Choose from the following categories:\n"
        "{top20_labels}\n"
        "Only the category name, no need to explain.\n"
        "For example, if the image is a sun flower, you should answer: sun flower."
    )
    return prompt

def get_prompt_list(data, prompt):
    prompt_list = []
    for item in data:
        top20_labels = item['top20_labels']
        top20_labels_str = '['+','.join(top20_labels)+']'
        new_prompt = prompt.format(top20_labels=top20_labels_str)
        prompt_list.append(new_prompt)
    print(f"Generated {len(prompt_list)} prompts.")
    print("Example:\n",prompt_list[0])
    return prompt_list

def construct_requests(clip_data,prompt_list, output_jsonl_path):
    requests = []

    for idx, item in enumerate(clip_data):
        image_path = item["image_path"]
        label_name = item["label_name"]
        prompt = prompt_list[idx]
        request = format_request(
            custom_id=str(idx),
            system_prompt="You are a helpful assistant who specializes in fine-grained classification.",
            user_prompt=prompt,
            image_path=image_path,
            temperature=0.25,
            max_tokens=4096,
        )
        requests.append(request)

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
    print(f"Saved {len(requests)} requests.")
    return output_jsonl_path

def chunked_batch_process(
    clip_data,
    prompt_list,
    jsonl_output_path,
    result_path,
    base_url="http://127.0.0.1:12345/v1",
    batch_size=500,
    intermediate_folder="./intermediate_results",
    dataset_name="dataset"
):

    assert len(clip_data) == len(prompt_list)
    processor = OpenAIBatchProcessor(base_url=base_url, api_key="EMPTY")

    all_output = []

    for i in range(0, len(clip_data), batch_size):
        sub_clip_data = clip_data[i:i+batch_size]
        sub_prompt_list = prompt_list[i:i+batch_size]
        batch_index = i // batch_size

        sub_jsonl_path = jsonl_output_path.replace(".jsonl", f"_{batch_index}.jsonl")
        intermediate_json_path = os.path.join(intermediate_folder, f"{dataset_name}_{batch_index}.json")

        # Skip if intermediate result already exists
        if os.path.exists(intermediate_json_path):
            with open(intermediate_json_path, 'r') as f:
                batch_output = json.load(f)
            if len(batch_output)< batch_size and len(batch_output) != len(sub_clip_data):
                print(f"[Batch {batch_index}] Existing intermediate result is incomplete. Reprocessing...")
            else:
                print(f"[Batch {batch_index}] Intermediate result already exists. Skipping...")
                all_output.extend(batch_output)
                continue

        print(f"[Batch {batch_index}] Sending {len(sub_clip_data)} requests...")
        construct_requests(sub_clip_data, sub_prompt_list, sub_jsonl_path)

        responses = processor.process_batch(sub_jsonl_path, endpoint="/v1/chat/completions", completion_window="24h")

        batch_output = []
        for item, resp in zip(sub_clip_data, responses):
            caption = resp["response"]["body"]["choices"]["message"]["content"]
            batch_output.append({
                "image_path": item["image_path"],
                "true_label": item["label_name"],
                "top20_labels": item["top20_labels"],
                "lmm_pred_with_top20": caption
            })
        save_json(batch_output, intermediate_json_path)
        print(f"[Batch {batch_index}] Saved intermediate results, total {len(batch_output)} samples.")

        all_output.extend(batch_output)

    save_json(all_output, result_path)
    print(f"Saved final complete results, total {len(all_output)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculating Backbone")
    parser.add_argument("--dataset", type=str, default="dog", help="dataset name (flower, cub, food, pet, aircraft, dog, car, sun)")
    parser.add_argument("--model", type=str, default="clip-rn50",  help="backbone (clip-b16, clip-rn50, siglip)")
    parser.add_argument("--use_ensemble", type=int, default=0, help="use ensemble prompts, 1 for yes, 0 for no (default)")
    parser.add_argument("--topk", type=int, default=20, help="number of top predictions to save")
    parser.add_argument("--lmm", type=str, default="Qwen2.5-VL-32B", help="LMM")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Processing dataset: ", args.dataset)
    print("Backbone model: ", args.model)
    print("Use ensemble prompts: ", bool(args.use_ensemble))
    dataset_dir, class_names = dataset_preprocess(args.dataset)

    sample_path = f"./test_sample/{args.dataset}_test.json"
    if args.use_ensemble:
        backbone_data_path = f"./backbone/{args.model}-ensemble/{args.dataset}.json"
    else:
        backbone_data_path = f"./backbone/{args.model}/{args.dataset}.json"


    test_samples = load_json(sample_path)
    print("Size of test dataset: ", len(test_samples))

    backbone_data = load_json(backbone_data_path)
    print("Loading backbone data from: ", backbone_data_path, "Size of data: ", len(backbone_data))

    prompt = get_prompt()
    print(f"prompt:\n {prompt}")


    prompt_list = get_prompt_list(backbone_data, prompt)
    if args.use_ensemble:
        request_save_path = f"./image_captions/{args.lmm}/conclusion/{args.model}-ensemble/requests/{args.dataset}_request.jsonl"
        lmm_infer_path = f"./image_captions/{args.lmm}/conclusion/{args.model}-ensemble/{args.dataset}.json"
        intermediate_folder = f"./image_captions/{args.lmm}/conclusion/{args.model}-ensemble/intermediate"
    else:
        request_save_path = f"./image_captions/{args.lmm}/conclusion/{args.model}/requests/{args.dataset}_request.jsonl"
        lmm_infer_path = f"./image_captions/{args.lmm}/conclusion/{args.model}/{args.dataset}.json"
        intermediate_folder = f"./image_captions/{args.lmm}/conclusion/{args.model}/intermediate"

    print(backbone_data[0])

    chunked_batch_process(
        clip_data=backbone_data,
        prompt_list=prompt_list,
        jsonl_output_path=request_save_path,
        result_path=lmm_infer_path,
        base_url="http://192.168.10.102:12345/v1",
        batch_size=500,
        intermediate_folder=intermediate_folder,
        dataset_name=args.dataset
    )


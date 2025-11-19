import json
import os.path
import time
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from utils import *
from dataset_split import dataset_preprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_dataset_type(dataset_name):
    if dataset_name == "cub":
        return "bird"
    elif dataset_name == "sun":
        return "scene"
    else:
        return dataset_name

def get_image_descriptions(dataset_name,backbone,feature_data, conclusion_data,save_description=False):
    image_descriptions = []
    dataset_type = get_dataset_type(dataset_name)

    for item1, item2 in zip(feature_data, conclusion_data):
        if item1["image_path"] == item2["image_path"]:
           feature_description = item1["feature_description"]
           lmm_conclusion = item2["lmm_pred_with_top20"]
           image_description = f"{feature_description}\n【Conclusion】The {dataset_type} in the image may be {lmm_conclusion}."
           image_descriptions.append({
               "image_path": item1["image_path"],
               "true_label": item1["true_label"],
               "image_description": image_description,
           })
        else:
            print(f" Loading the unmatched data.")
            return
    if save_description:
        save_path = f"./image_descriptions/{backbone}/{dataset_name}_descriptions.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_json(image_descriptions, save_path)
        print("Save the image descriptions to", save_path)

    return image_descriptions

def format_instruction(datasetType,instruction, imageDescription, categoryDescription):
    output = (f"<Instruct>:{instruction}\n"
              f"<{datasetType.capitalize()} Description>:{imageDescription}\n"
              f"<Category Description>:{categoryDescription}")
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return true_vector.tolist(), false_vector.tolist(), scores


def text_reranker(dataset_name, image_descriptions,backbone_data,save_dir):
    correct = 0
    records = []
    jsonl_path = os.path.join(save_dir,f"{dataset_name}_stream.jsonl")
    final_json_path = os.path.join(save_dir,f"{dataset_name}.json")

    processed_images = set()
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                record = json.loads(line)
                records.append(record)
                if record["true_label"]==record["pred_label"]:
                    correct += 1
                processed_images.add(record["image_path"])
            print("Loaded processed records:", len(records))
    except FileNotFoundError:
        pass

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a") as jsonl_file:
        for i, data in tqdm(enumerate(image_descriptions), total=len(image_descriptions), desc=f"Evaluating {dataset_name}"):
            image_path = data["image_path"]
            true_label = data["true_label"]
            image_description = data["image_description"]

            if image_path in processed_images:
                continue

            backbone_path = backbone_data[i]["image_path"]
            candidate_categories = backbone_data[i]["top20_labels"]
            candidate_index = backbone_data[i]["top20_index"]

            try:
                category_descriptions = [class_dict[cat] for cat in candidate_categories]
            except KeyError as e:
                print(f"[Warning] Category {e} not found in class_dict. Skipping.")
                continue

            dataset_type = get_dataset_type(dataset_name)
            task = f"Given a <{dataset_type.capitalize()} Description> and a <Category Description>, determine if the {dataset_type} description belongs to this category."

            pairs = [
                format_instruction(dataset_type, task, image_description, cat_desc)
                for cat_desc in category_descriptions
            ]

            try:
                inputs = process_inputs(pairs)
                true_vector, false_vector, scores = compute_logits(inputs)
            except RuntimeError as e:
                print(f"[RuntimeError] Skipping {image_path}: {e}")
                torch.cuda.empty_cache()
                return

            top_index = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[0]
            pred_label = class_names[candidate_index[top_index]]

            if pred_label == true_label:
                correct += 1

            record = {
                "image_path": image_path,
                "true_label": true_label,
                "pred_label": pred_label,
                "true_vector": true_vector,
                "false_vector": false_vector,
                "scores": scores
            }
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()
            records.append(record)

    acc = correct / len(records)
    print(f"Reranker Acc: {acc:.4f}")

    save_json(records, final_json_path)
    print("Successfully save the results to ", final_json_path)

if __name__ == "__main__":

    set_seed(42)

    parser = argparse.ArgumentParser(description="Text Reranker ")
    parser.add_argument("--dataset", type=str, default="cub", help="dataset name (flower, cub, food, pet, aircraft, dog, car, sun)")
    parser.add_argument("--backbone", type=str, default="clip-b16",  help="backbone (clip-b16, clip-rn50)")
    parser.add_argument("--use_ensemble", type=int, default=0, help="Use prompt ensemble method in clip [0: no (default), 1: yes]")
    parser.add_argument("--reranker_model", type=str, default="Qwen3-Reranker-8B", help="Qwen3-Reranker-8B(default), Qwen3-Reranker-4B, Qwen3-Reranker-0.6B")
    parser.add_argument("--lmm", type=str, default="Qwen2.5-VL-32B", help="Qwen2.5-VL-32B(default),Qwen2.5-VL-7B")
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir, class_names = dataset_preprocess(dataset_name)
    dataset_type = get_dataset_type(dataset_name)
    print("Dataset：", dataset_name," Type：", dataset_type," Category Number：", len(class_names))

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = cfg['model']['reranker']
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        ).eval()
    token_false_id = tokenizer.convert_tokens_to_ids('no')
    token_true_id = tokenizer.convert_tokens_to_ids('yes')
    max_length = 4096


    if args.use_ensemble:
        backbone_data_path = f"./backbone/{args.backbone}-ensemble/{args.dataset}.json"
        conclusion_data_path = f"./image_captions/{args.lmm}/conclusion/{args.backbone}-ensemble/{args.dataset}.json"
        reranker_save_path = f"./text_reranker/{args.reranker_model}/{args.backbone}-ensemble"
    else:
        backbone_data_path = f"./backbone/{args.backbone}/{args.dataset}.json"
        conclusion_data_path = f"./image_captions/{args.lmm}/conclusion/{args.backbone}/{args.dataset}.json"
        reranker_save_path = f"./text_reranker/{args.reranker_model}/{args.backbone}"
    feature_description_path = f"./image_captions/{args.lmm}/feature_description/{args.dataset}.json"
    category_path = f"./category_descriptions/{args.dataset}.xlsx"

    backbone_data = load_json(backbone_data_path)
    feature_data = load_json(feature_description_path)
    conclusion_data = load_json(conclusion_data_path)

    image_descriptions = get_image_descriptions(dataset_name,args.backbone,feature_data,conclusion_data,save_description=False)
    print(f"Successfully load {len(image_descriptions)} image captions.")
    print(f"For example: \n {image_descriptions[0]}")

    df = pd.read_excel(category_path)
    class_dict = dict(zip(df['category'],df['description']))

    prefix = (f"<|im_start|>system\n "
              f"Judge whether the {dataset_type} described in <{dataset_type.capitalize()} Description> belongs to the category described in <Category Description>. "
              f"Focus on comparing the features in the <{dataset_type.capitalize()} Description> and <Category Description> carefully to make your judgment."
              f"Respond with only \"yes\" if the two refer to the same {dataset_type} species, otherwise respond \"no\"."
              f"<|im_end|>\n<|im_start|>user\n"
              )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    task = f"Given a <{dataset_type.capitalize()} Description> and a <Category Description>, determine if the {dataset_type} description belongs to this category."
    text_reranker(dataset_name,image_descriptions,backbone_data,reranker_save_path)


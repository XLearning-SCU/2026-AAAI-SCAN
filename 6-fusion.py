import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset_split import dataset_preprocess
from utils import set_seed, load_json, save_json
import torch.nn.functional as F

#Normalize
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val == 0:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)

#Compute Dirichlet parameters
def compute_dirichlet_params(score):
    return score + 1

def compute_pred_prob(alpha):
    S = np.sum(alpha)
    return alpha/S

def compute_uncertainty(alpha):
    K = len(alpha)
    S = np.sum(alpha)
    return K/S

def dynamic_fusion(clip_data, reranker_data, topk=20, temperature = 40, save_path = ""):
    is_text_correct = 0
    correct = 0
    correct_top3 = 0
    records = []

    for i, item in enumerate(clip_data):
        image_path = item["image_path"]
        true_label = item["label_name"]

        text_pred_label = reranker_data[i]["pred_label"]
        if text_pred_label == true_label:
            is_text_correct += 1

        top20_clip_similarity = np.array(item["top20_prob"][:topk], dtype=np.float32)

        top1 = top20_clip_similarity[0]
        margins = top1 - top20_clip_similarity[1:]

        margins = margins* temperature

        alpha = compute_dirichlet_params(margins)
        uncertainty = compute_uncertainty(alpha)

        a = 1-uncertainty
        text_scores = np.array(reranker_data[i]["scores"][:topk], dtype=np.float32)

        clip_norm = normalize(top20_clip_similarity)
        text_norm = normalize(text_scores)

        combined_score = a * clip_norm + (1-a) * text_norm

        max_index = int(np.argmax(combined_score))
        best_label = item["top20_labels"][max_index]

        top3_indices = np.argsort(combined_score)[-3:][::-1]
        top3_labels = [item["top20_labels"][idx] for idx in top3_indices]

        if best_label == true_label:
            correct += 1
        if true_label in top3_labels:
            correct_top3 += 1

        records.append({
            "image_path": image_path,
            "true_label": true_label,
            "fusion_pred_label": best_label,
            "clip_norm": clip_norm.tolist(),
            "text_norm": text_norm.tolist(),
            "combined_score": combined_score.tolist(),
        })
    save_json(records, save_path)
    text_acc = is_text_correct / len(records)
    fusion_top1_acc = correct / len(records)
    fusion_top3_acc = correct_top3 / len(records)
    return text_acc, fusion_top1_acc, fusion_top3_acc


if __name__ == "__main__":

    set_seed(42)

    parser = argparse.ArgumentParser(description="Text Reranker ")
    parser.add_argument("--dataset", type=str, default="flower", help="dataset name (flower, cub, food, pet, aircraft, dog, car, sun)")
    parser.add_argument("--backbone", type=str, default="clip-b16",  help="backbone (clip-b16, clip-rn50, siglip)")
    parser.add_argument("--use_ensemble", type=int, default=0, help="Use prompt ensemble method in clip [0: no (default), 1: yes]")
    parser.add_argument("--reranker_model", type=str, default="Qwen3-Reranker-8B", help="Qwen3-Reranker-8B(default), Qwen3-Reranker-4B, Qwen3-Reranker-0.6B")
    parser.add_argument("--lmm", type=str, default="Qwen2.5-VL-32B", help="Qwen2.5-VL-32B(default),Qwen2.5-VL-7B")
    args = parser.parse_args()

    dataset_dir, class_names = dataset_preprocess(args.dataset)
    if args.use_ensemble:
        backbone_path = f"./backbone/{args.backbone}-ensemble/{args.dataset}.json"
        reranker_path = f"./text_reranker/{args.reranker_model}/{args.backbone}-ensemble/{args.dataset}.json"
        save_path = f"./fusion_results/{args.reranker_model}/{args.backbone}-ensemble/{args.dataset}.json"
    else:
        backbone_path = f"./backbone/{args.backbone}/{args.dataset}.json"
        reranker_path = f"./text_reranker/{args.reranker_model}/{args.backbone}/{args.dataset}.json"
        save_path = f"./fusion_results/{args.reranker_model}/{args.backbone}/{args.dataset}.json"

    backbone_data = load_json(backbone_path)
    reranker_data = load_json(reranker_path)

    print("Current dataset:", args.dataset)
    print("Loaded", len(backbone_data), "backbone samples from", backbone_path)
    print("Loaded", len(reranker_data), "reranker samples from", reranker_path)

    print("===========Dynamic Fusion============")
    t_list = [10,20,30,40,50]
    for temperature in t_list:
        print("temperature = ", temperature)
        text_acc, fusion_top1_acc, fusion_top3_acc = dynamic_fusion(backbone_data, reranker_data, topk=20, temperature=temperature, save_path=save_path)
        print(f"Reranker Acc:{text_acc:.4f}", f"Fusion Top1 Acc:{fusion_top1_acc:.4f}", f"Fusion Top3 Acc:{fusion_top3_acc:.4f}")
        print("==================================")


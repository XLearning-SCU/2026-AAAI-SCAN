import clip
from dataset_split import dataset_preprocess
from utils import *
from PIL import Image
import argparse
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

#basic prompt
def get_clip_prompts(class_names):
    prompts = [f"a photo of a {class_name}." for class_name in class_names]
    return prompts

#official ensemble prompt：https://github.com/openai/CLIP/blob/main/data/prompts.md
def get_clip_prompts_ensemble(class_names, dataset_name):
    if dataset_name == "flower":
        prompt_templates = ['a photo of a {}, a type of flower.']
    elif dataset_name == "cub":
        prompt_templates = ['a photo of a {}, a type of bird.']
    elif dataset_name == "food":
        prompt_templates = ['a photo of {}, a type of food.']
    elif dataset_name == "pet":
        prompt_templates = ['a photo of a {}, a type of pet.']
    elif dataset_name == "aircraft":
        prompt_templates = [
            'a photo of a {}, a type of aircraft.',
            'a photo of the {}, a type of aircraft.'
        ]
    elif dataset_name == "dog":
        prompt_templates = ['a photo of a {}, a type of dog.']
    elif dataset_name == "car":
        prompt_templates = [
            'a photo of a {}.',
            'a photo of the {}.',
            'a photo of my {}.',
            'i love my {}!',
            'a photo of my dirty {}.',
            'a photo of my clean {}.',
            'a photo of my new {}.',
            'a photo of my old {}.'
        ]
    elif dataset_name == "sun":
        prompt_templates = [
            'a photo of a {}.',
            'a photo of the {}.'
        ]
    else:
        print("Unknown dataset.")
        return
    all_prompts = []
    for cls in class_names:
        prompts_for_cls = [template.format(cls) for template in prompt_templates]
        all_prompts.append(prompts_for_cls)
    return all_prompts

def get_text_features_ensemble(model, prompts_nested):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_features = []
    for prompts in prompts_nested:
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
        avg_embedding = embeddings.mean(dim=0)
        avg_embedding /= avg_embedding.norm()
        text_features.append(avg_embedding)
    return torch.stack(text_features)

def calculate_clip(model, preprocess, dataset_name, save_topk, sample_path, save_path, use_ensemble=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir, class_names = dataset_preprocess(dataset_name)
    if use_ensemble:
        prompts = get_clip_prompts_ensemble(class_names, dataset_name)
        text_features = get_text_features_ensemble(model, prompts)
    else:
        prompts = get_clip_prompts(class_names)
        text_token = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_token)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    total = 0
    results_per_image = []
    correct_at_k = {k: 0 for k in [1, 5, 10, 20]}

    data = load_json(sample_path)

    for item in tqdm(data):
        total += 1
        label_index = item["label_index"]
        image = preprocess(Image.open(item['image_path'])).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T
            similarity = similarity.squeeze(0)

        for k in correct_at_k:
            if label_index in similarity.topk(k).indices.tolist():
                correct_at_k[k] += 1

        top_k_indices = similarity.topk(save_topk).indices.tolist()
        top_k_similarity = similarity.topk(save_topk).values.tolist()
        top_k_labels = [class_names[index] for index in top_k_indices]
        pred_label = top_k_labels[0]

        results_per_image.append({
            "image_path": item['image_path'],
            "label_index": item['label_index'],
            "label_name": item['label_name'],
            "pred_label": pred_label,
            "backbone_prob": similarity.tolist(),
            "top20_index": top_k_indices,
            "top20_prob": top_k_similarity,
            "top20_labels": top_k_labels,
        })

    print("Total samples:", total)
    acc_results = [[k, correct_at_k[k] / total] for k in correct_at_k]
    for k, accuracy in acc_results:
        print(f"Top-{k}: {accuracy:.4f}")

    save_json(results_per_image, save_path)
    print("Save the result successfully to: ", save_path)
    return acc_results

if __name__ == "__main__":

    cfg = load_config()

    parser = argparse.ArgumentParser(description="Calculating Backbone")
    parser.add_argument("--dataset", type=str, default="cub", help="dataset name (flower, cub, food, pet, aircraft, dog, car, sun)")
    parser.add_argument("--model", type=str, default="clip-b16",  help="backbone (clip-b16, clip-rn50, siglip)")
    parser.add_argument("--use_ensemble", type=int, default=0, help="use ensemble prompts, 1 for yes, 0 for no (default)")
    parser.add_argument("--topk", type=int, default=20, help="number of top predictions to save")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "clip-b16":
        model_name = "ViT-B/16"
        model, preprocess = clip.load(model_name, device=device)
    elif args.model == "clip-rn50":
        model_name = "RN50"
        model, preprocess = clip.load(model_name, device=device)
    else:
        print("Unknown model.")
        exit(1)

    print("Loading Model:", model_name)

    dataset_name = args.dataset
    test_data_path = f"{cfg['dataset']['test_path']}/{dataset_name}_test.json"
    print("Loading Dataset:", dataset_name, "from:", test_data_path)

    if args.use_ensemble:
        save_path = f"./backbone/{args.model}-ensemble/{args.dataset}.json"
    else:
        save_path = f"./backbone/{args.model}/{args.dataset}.json"

    calculate_clip(model, preprocess, dataset_name, args.topk, test_data_path, save_path, use_ensemble=args.use_ensemble)
    print("------------------------------")

import argparse
from openai import OpenAI
from dataset_split import dataset_preprocess
from utils import *


def format_request(
    custom_id,
    system_prompt,
    user_prompt,
    end_point="/v1/chat/completions",
    max_tokens=2048
):
    payload = {
        "custom_id": str(custom_id),
        "method": "POST",
        "url": end_point,
        "body": {
            "model": "gpt-4.1-mini-2025-04-14",
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    }
    return payload

def get_prompt_list(dataset_name, prompt_index, prompt_save_path):
    dataset_dir, class_names = dataset_preprocess(dataset_name)
    prompt_list = []

    with open("./prompts/category_descriptions_prompt.json") as f:
        data = json.load(f)

    prompt_template = data[prompt_index][dataset_name]

    for class_name in class_names:
        formatted_name = f"[{class_name}]"
        new_prompt = prompt_template.format(class_name=formatted_name)
        prompt_list.append(new_prompt)

    save_json(prompt_list, prompt_save_path)
    print(f"Saved {len(prompt_list)} prompts.")
    return prompt_list


def construct_request(dataset_name, prompt_list, request_save_path):
    _, class_names = dataset_preprocess(dataset_name)
    request_list = []

    for class_name, prompt in zip(class_names, prompt_list):

        custom_id = class_name
        system_prompt = (
            "You are an expert in fine-grained recognition, capable of "
            "identifying key discriminative features between closely related categories."
        )

        request = format_request(
            custom_id=custom_id,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=2048,
        )

        request_list.append(request)

    os.makedirs(os.path.dirname(request_save_path), exist_ok=True)
    with open(request_save_path, "w", encoding="utf-8") as f:
        for req in request_list:
            f.write(json.dumps(req) + "\n")

    print(f"Saved {len(request_list)} requests to {request_save_path}.")


if __name__ == "__main__":

    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Generate category descriptions using the OpenAI API (batch mode)"
    )
    parser.add_argument("--dataset", default="flower", type=str, help="Dataset name")
    args = parser.parse_args()

    dataset_name = args.dataset

    prompt_save_path = f"./category_descriptions/prompt_list/{dataset_name}_prompt_list.json"
    prompt_list = get_prompt_list(dataset_name, 0, prompt_save_path)


    request_save_path = f"./category_descriptions/request_list/{dataset_name}_request_list.jsonl"
    construct_request(dataset_name, prompt_list, request_save_path)

    client = OpenAI(api_key=cfg['openai']['api_key'])

    batch_input_file = client.files.create(
        file=open(request_save_path, "rb"),
        purpose="batch",
    )
    print("Uploaded batch file:", batch_input_file)

    client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"{dataset_name} category description"},
    )
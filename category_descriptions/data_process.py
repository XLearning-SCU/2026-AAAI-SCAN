import json
import pandas as pd

if __name__ == "__main__":
    dataset_list = ["flower", "cub", "food", "pet", "aircraft", "dog", "car", "sun"]

    for dataset_name in dataset_list:
        jsonl_path = f"./gpt_output/{dataset_name}.jsonl"
        data = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                custom_id = obj.get("custom_id", "")
                # Extract content field
                content = (
                    obj.get("response", {})
                    .get("body", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                data.append({"category": custom_id, "description": content})

        df = pd.DataFrame(data)
        excel_path = f"{dataset_name}.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"Exported {len(df)} records to {excel_path}")

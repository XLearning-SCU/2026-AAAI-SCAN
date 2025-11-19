import base64
import json
import time
from typing import Dict, Any

import openai
from easydict import EasyDict
from openai.types.chat import ChatCompletion

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_request(custom_id, system_prompt, user_prompt, image_path = None, end_point = '/v1/chat/completions',
                   temperature = 0.2,
                   max_tokens = 2048):
    user_content = []
    if image_path is not None:
        if isinstance(image_path, list):
            base64_images = [encode_image(x) for x in image_path]
            image_content = [{
                "type":"image_url",
                "image_url":{
                    "url":f"data:image/png;base64,{base64_image}"
                },
                "modalities":"multi-images",
            } for base64_image in base64_images]
            user_content.extend(image_content)
        else:
            base64_image = encode_image(image_path)
            image_content = {
                "type":"image_url",
                "image_url":{
                    "url":f"data:image/png;base64,{base64_image}",
                }
            }
            user_content.append(image_content)
    user_content.append({
        "type":"text",
        "text":user_prompt
    })
    payload = {
        "custom_id":str(custom_id),
        "method":"POST",
        "url":end_point,
        "body":{
            "model":"default",
            "temperature":temperature,
            "max_tokens":max_tokens,
            "messages":[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role":"user",
                    "content":user_content
                }
            ]
        }
    }
    # print(payload)
    return payload

def chat_completion_to_dict(chat_completion: ChatCompletion) -> Dict[str, Any]:
    return {
        "id":chat_completion.id,
        "choices":{
            "finish_reason":chat_completion.choices[0].finish_reason,
            "index":chat_completion.choices[0].index,
            "logprobs":chat_completion.choices[0].logprobs,
            "message":{
                "content":chat_completion.choices[0].message.content,
                "role":chat_completion.choices[0].message.role
            },
            "matched_stop":chat_completion.choices[0].matched_stop
        },
        "created":chat_completion.created,
        "model":chat_completion.model,
        "object":chat_completion.object,
        "usage":{
            "completion_tokens":chat_completion.usage.completion_tokens,
            "prompt_tokens":chat_completion.usage.prompt_tokens,
            "total_tokens":chat_completion.usage.total_tokens
        }
    }

class OpenAIBatchProcessor:
    def __init__(self, base_url = "http://127.0.0.1:30000/v1", api_key = "EMPTY"):
        # client = OpenAI(api_key=api_key)
        self.client = openai.Client(base_url = base_url, api_key = api_key)

    def process_batch(self, input_file_path, endpoint, completion_window, interval: int = 10):

        # Upload the input file
        with open(input_file_path, "rb") as file:
            if sum(1 for line in file) == 1:
                file.seek(0)
                result = [self.process_single(file.readlines())]
                return result
            file.seek(0)
            uploaded_file = self.client.files.create(file = file, purpose = "batch")

        # Create the batch job
        batch_job = self.client.batches.create(
                input_file_id = uploaded_file.id,
                endpoint = endpoint,
                completion_window = completion_window,
        )

        # Monitor the batch job status
        start_time = time.time()
        try_time = 0
        while batch_job.status not in ["completed", "failed", "cancelled"]:
            try_time += 1
            time.sleep(interval)  # Wait for few seconds before checking the status again
            if try_time % 10 == 0:
                print(f"Batch job status: {batch_job.status}...trying again every {interval} seconds...")
            batch_job = self.client.batches.retrieve(batch_job.id)

        # Check the batch job status and errors
        if batch_job.status == "failed":
            print(f"Batch job failed with status: {batch_job.status}")
            print(f"Batch job errors: {batch_job.errors}")
            return None

        # If the batch job is completed, process the results
        if batch_job.status == "completed":
            cur_time = int(time.time() - start_time)
            # print result of batch job
            print("batch finished using {} sec:".format(cur_time), batch_job.request_counts)

            result_file_id = batch_job.output_file_id
            # Retrieve the file content from the server
            file_response = self.client.files.content(result_file_id)
            result_content = file_response.read()  # Read the content of the file

            # Save the content to a local file
            result_file_name = f"batch_job_chat_results.jsonl"
            with open(result_file_name, "wb") as file:
                file.write(result_content)  # Write the binary content to the file
            # Load data from the saved JSONL file
            results = []
            with open(result_file_name, "r", encoding = "utf-8") as file:
                for line in file:
                    json_object = json.loads(
                            line.strip()
                    )  # Parse each line as a JSON object
                    results.append(json_object)

            return results
        else:
            print(f"Batch job failed with status: {batch_job.status}")
            return None

    def process_single(self, request):
        request = json.loads(request[0].decode('utf-8').strip("[]\n"))
        custom_id = request["custom_id"]
        response = self.client.chat.completions.create(**request["body"])
        response = chat_completion_to_dict(response)
        response = {
            "id":"0",
            "custom_id":custom_id,
            "response":{
                "body":response
            }
        }
        return response

# if __name__ == "__main__":
#     client = OpenAIBatchProcessor("192.168.49.59:12345/v1")
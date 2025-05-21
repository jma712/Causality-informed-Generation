import json
import os
import anthropic

import time
import base64
from mimetypes import guess_type

import base64
import httpx
import json
import os
from info import scene
from PIL import Image

import random
import os
import shutil
import numpy as np
import pandas as pd
import concurrent.futures
from functools import partial
import ast
import re



client = anthropic.Anthropic(api_key=xxx)

def random_select_images(dir_path, num_images):
    image_list = os.listdir(dir_path)
    random.shuffle(image_list)
    # dir + image name
    image_paths = [os.path.join(dir_path, image) for image in image_list[:num_images]]

    return image_paths


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def claude_eval(input_prompt,image_paths,system_prompt=None,model_id="claude-3-5-sonnet-20241022",temperature=1,max_tokens=1024):
    if type(image_paths) != list:
        image_paths=ast.literal_eval(image_paths)

    if len(image_paths) > 10:
        raise ValueError("You can input a maximum of 10 images at once.")
    
    # Create the image payload for each image
    image_messages = []
    for image_path in image_paths:
        data_url = local_image_to_data_url(image_path)
        image_media_type = "image/png"  # Assuming all images are JPEG. Adjust if needed.
        image_data = data_url.split(",")[1]

        # Add each image as an input message
        image_messages.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_data,
            },
        })

    # Combine images with the input text prompt
    messages = image_messages + [{
        "type": "text",
        "text": input_prompt
    }]

    if system_prompt is None:
        response = None
        for _ in range(len(image_paths)):  # Retry mechanism
            try:
                response = client.messages.create(
                                                    model=model_id,
                                                    max_tokens=max_tokens,
                                                    temperature=temperature,
                                                    messages=[
                                                        {
                                                            "role": "user",
                                                            "content": messages
                                                        }
                                                    ],
                                                )
                
                if response is not None:
                    break
            except Exception as e:
                print(f"[CLAUDE3 ERROR]: {e}")
                time.sleep(5)
                response = None

        if response is not None:
            return response.content[0].text
        
    else:
        response = None
        for _ in range(10):  # Retry mechanism
            try:
                response = client.messages.create(
                            model=model_id,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system= [
                                        {
                                            "type": "text",
                                            "text": system_prompt,
                                        }
                                    ],

                            messages=[
                                        {
                                            "role": "user",
                                            "content": messages
                                        }
                                    ],
                                                )
                
                if response is not None:
                    break
            except Exception as e:
                print(f"[CLAUDE3 ERROR]: {e}")
                time.sleep(5)
                response = None

        if response is not None:
            return response.content[0].text

    return None



def compose_content(dict_info):
  num_of_v = len(dict_info["variables"])
  variables = dict_info["variables"]
  content = ''
  for i,v in enumerate(variables):
    content += f"{i+1}. {variables[v]}\n"
  content = f"There are {num_of_v} variables: \n{content}.\n" 
  content += "Please fill this causality adjacency matrix:\n"
  return content


def prompt_composition(scene_info_dict):
  matrix = scene_info_dict['adjacency_matrix']
  matrix = str(matrix).replace("1", "_,").replace("0", "_,").replace("_,]", '_]')
  matrix = matrix.replace("_]", "_],")
  matrix = '```\n' + matrix
  matrix += '\n```'
  matrix_info = ".\nIn the matrix, matrix[i][j] = 1 means variable i causes variable j, matrix[i][j] = 0 means there is not direct causal relationship."
  scene_info = compose_content(scene_info_dict)
  prompt = scene_info + matrix + matrix_info
  return prompt


def extract_answer(text):
    # 使用正则表达式匹配 "The answer is: ..." 的内容
    match = re.search(r"The answer is:\s*(.*)", text)
    if match:
        return match.group(1).strip()[0]  # 返回匹配的答案并去除多余空格
    return None  # 如果没有匹配到，返回 None



def main(data):
    image_paths = data['images']

    # 获取 prompt
    prompt = data['message']
    system_info = None

    # 获取 response
    response = claude_eval(prompt, image_paths, system_info)
    data['claude_response'] = response
    data['claude_answer'] = extract_answer(response)
    return data

def process_row(row):
    return main(row)

if __name__ == "__main__":
    csv_dirs = 'intevention'
    result_dir = 'intevention_results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for scene_name in os.listdir(csv_dirs):
        scene_path = os.path.join(csv_dirs, scene_name)

        # 加载数据
        data = pd.read_csv(scene_path)
        name = scene_name.split('.')[0]

        # 数据长度
        length = len(data)
        #responses 是需要存为一个csv文件
        responses = []

        # 使用 ThreadPoolExecutor 进行多线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            # 提交任务并收集 Future 对象
            rows = [data.iloc[i] for i in range(length)]  # 获取每一行数据
            futures = [executor.submit(process_row, row) for row in rows]

            # 等待任务完成并获取结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    data = future.result()
                    print(data['claude_response'], data['claude_answer'])
                    responses.append(data)
                except Exception as e:
                    print(f"Error occurred: {e}")

        # 保存 responses
        responses = pd.DataFrame(responses)
        responses.to_csv(os.path.join(result_dir, f"{name}_responses.csv"), index=False)


    print("All done!")



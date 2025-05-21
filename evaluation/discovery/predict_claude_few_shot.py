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


client = anthropic.Anthropic(api_key=API_KEY)

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
        for _ in range(10):  # Retry mechanism
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


def get_few_shot_samples(all_scenes,scene_name):
    few_shot_samples=[]
    for key in all_scenes:
        if key!=scene_name:
            #if all_scenes[key] has attribute sample_result
            if "sample_result" in all_scenes[key].keys():

                few_shot_samples.append(all_scenes[key]["sample_result"])

    return few_shot_samples

def get_few_shot_prompt(few_shot_samples,num_samples=3):
    #randomly select num_samples from few_shot_samples, max= 4, or you can add samples into info.py
    few_shot_samples=random.sample(few_shot_samples,num_samples)

    #create prompt
    prompt=""
    i=1
    for sample in few_shot_samples:
        prompt+= f"Example {i}:"
        prompt+=sample
        prompt+="\n"
        i+=1

    prompt= prompt+ "Based on the examples above, answer the following questions:\n"
    return prompt



def main(save_name):
    scene_info= scene()
    all_scenes=scene_info.get_all_scenes()

    responses={}

    for key,value in all_scenes.items():
        scene_name= key
        #get images to be evaluated
        dir_path=value["file_name"]
        image_paths = random_select_images(dir_path, 10)

        #get prompt
        few_shot_samples=get_few_shot_samples(all_scenes,scene_name)
        few_shot_prompt=get_few_shot_prompt(few_shot_samples)

        scene_info_dict = scene_info.scenes[scene_name]
        prompt = prompt_composition(scene_info_dict)
        system_info = "Analyze the provided images and identify causal relationships between the variables. Complete the causality adjacency matrix based on the identified relationships and briefly explain your conclusions."

        prompt= few_shot_prompt + prompt
        print(prompt)

        #get response
        response = claude_eval(prompt,image_paths,system_info)
        print(response)
        responses[scene_name]=response

    #save responses into json indent 4
    with open(save_name, 'w') as f:
        json.dump(responses, f, indent=4)


if __name__ == "__main__":
    for i in range(9):
        save_name = "responses_3_shot(ICL)_"+str(i)+".json"
        main(save_name)
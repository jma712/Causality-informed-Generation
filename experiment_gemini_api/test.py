import json
import os
import anthropic
from anthropic import AnthropicVertex

import time
#from nltk.tokenize import word_tokenize
import httpx
import base64
from mimetypes import guess_type
import argparse
# from instruction_generation_yesbut_1 import formulate_instruction
# from instruction_generation_yesbut_1 import *



PROJECT_ID = "mimetic-kit-445917-d8"
# LOCATION = ""  # @param {type:"string"}
projectid = PROJECT_ID
LOCATION = "us-east4"

client = AnthropicVertex(region=LOCATION, project_id=projectid)



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


def claude3_vision_generation(input_prompt, image_path, model="claude-3-opus-20240229", temperature=1):
    

    data_url = local_image_to_data_url(image_path)
    image_media_type="image/jpeg"
    image_data = data_url.split(",")[1] 


    for _ in range(10):
        try:
            response = client.messages.create(model="claude-3-opus@20240229",
                                            max_tokens=512,
                                            temperature=temperature,
                                            messages=[
                                                {
                                                    "role": "user",
                                                    "content": [
                                                                {
                                                                    "type": "image",
                                                                    "source": {
                                                                        "type": "base64",
                                                                        "media_type": image_media_type,
                                                                        "data": image_data,
                                                                    },
                                                                },
                                                                {
                                                                    "type": "text",
                                                                    "text": input_prompt
                                                                }
                                                                ],
                                                }
                                            ],
                                            )

            
            if response is not None:
                break
        except Exception as e:
            print(["[CLAUDE3 ERROR]: ", [e]])
            response = None
            time.sleep(5)
    if response != None:
        #print(response.content[0].text)
        response = response.content[0].text
    return response



respone=claude3_vision_generation("Generate a description of the image", "/home/lds/github/Causality-informed-Generation/database/rendered_image.png", model="claude-3-opus-20240229", temperature=1)



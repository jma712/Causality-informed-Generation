# import IPython
# from IPython.display import HTML, Markdown, display
import vertexai
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# app = IPython.Application.instance()
PROJECT_ID = "mimetic-kit-445917-d8"
LOCATION = "us-central1"  # @param {type:"string"}

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")

vertexai.init(project=PROJECT_ID, location=LOCATION)

from anthropic import AnthropicVertex
from google.auth import default, transport
import logging
import csv
from datetime import datetime
import openai
from vertexai.evaluation import (
    EvalTask,
    MetricPromptTemplateExamples,
    PairwiseMetric,
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
)
from vertexai.generative_models import GenerativeModel
import vertexai

from vertexai.generative_models import GenerativeModel, Part, Image

import logging
import warnings

import pandas as pd
import sys
import os
import random
from tqdm import tqdm
sys.path.append('/home/lds/github/Causality-informed-Generation/inference/evaluation')
from utils import info
from utils import evaluation


logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

model = GenerativeModel(
    "gemini-1.5-pro-002",
    generation_config={
      # "temperature": 0.6, 
      "max_output_tokens": 256, 
      "top_k": 1},
)

def find(text, target):
  return text[text.find(target)+ len(target)]

def main(paths):
  model = GenerativeModel(
      "gemini-1.5-pro-002",
      generation_config={
        # "temperature": 0.6, 
        "max_output_tokens": 1000, 
        "top_k": 1},
  )
  count_prediction = count_predictions("./convex_intervention.log")
  print(count_prediction)
  for path in paths:
    data = pd.read_csv(path)
    import ast
    if isinstance(data['images'][0], str):
        data['images'] = data['images'].apply(ast.literal_eval)

    # Convert to a list of paths
    images = data['images'].tolist()


    prompt = data['message'].tolist()
    gt = data['ground_truth'].tolist()
    count = 0
    for image, prompt, gt in (zip(images, prompt, gt)):
        if count < count_prediction:
          # print(count)
          # print("skip")
          count += 1
          continue
        count += 1
        print(image)
        print(prompt)
        print("gt:",gt)
        # print(generate_content(model, image, prompt))
        print("==================================")
        imgs = [Part.from_image(Image.load_from_file(path)) for path in image]
    
        prompt = imgs + [Part.from_text(prompt)]
  
        try:
            response = model.generate_content(prompt)
        except Exception as e:
            print("Model failed to generate content: %s", str(e))
            raise
        print(response.text)
        pre = find(response.text, "The answer is: ")
        print("prediction:",pre)
        print()
        print("==================================")
    # return response.text
    
    
paths = [
  # "/home/lds/github/Causality-informed-Generation/experiment_chatgpt_api/causal_OpenAI_res/Magnets_intervention_one_view/basic_synbackground_-1.csv",
  # "/home/lds/github/Causality-informed-Generation/experiment_chatgpt_api/causal_OpenAI_res/Reflection_intervention/basic_synbackground_-1.csv",
  "/home/lds/github/Causality-informed-Generation/experiment_chatgpt_api/causal_OpenAI_res/Convex_intervention/basic_synbackground_-1.csv"
  ]

import re

def count_predictions(path):
    """
    解析日志文件，统计 prediction 的出现次数。
    
    参数:
        path (str): 日志文件路径。
    
    返回:
        int: prediction 的总次数。
    """
    try:
        # 读取日志文件内容
        with open(path, "r") as file:
            log_content = file.read()
        
        # 使用正则表达式提取所有 prediction
        predictions = re.findall(r"prediction: (\w+)", log_content)
        
        # 返回 prediction 的次数
        return len(predictions)
    except FileNotFoundError:
        print(f"文件未找到: {path}")
        return 0
    except Exception as e:
        print(f"解析日志文件时发生错误: {e}")
        return 0


main(paths)
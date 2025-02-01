import os  
import random
import numpy as np  
import sys
import os
import random
from tqdm import tqdm
sys.path.append('/home/lds/github/Causality-informed-Generation/inference/evaluation/utils/')
from info import scene

class evaluation():
  def __init__(self):
    """
    reference: 
    [D'ya like DAGs? A Survey on Structure Learning and Causal Discovery](https://arxiv.org/pdf/2103.02582)
    """
    self.metrics = {"TPR": 0.0, "FPR": float("inf"), "SHD": float("inf")}
    abs_path = os.path.abspath(__file__)
    abs_dir_path = os.path.dirname(abs_path)
    print(abs_dir_path)
    self.scenes_imgs_database = {"reflection": f"{abs_dir_path}/../../../code1/database/rendered_reflection_128P",
                                 "magnetic": "../../../code1/database/rendered_magnetic_128P",
                                 "seesaw": "../../../code1/database/rendered_seesaw_128P",
                                 "spring": "../../../code1/database/rendered_spring_128P",}
    
    self.inference_res = None

  def get_inference_result(self, scene_name, imgs_num = 10, random = 0):
    imgs = self.get_scene_img(scene_name, imgs_num, random)
    return imgs

  def get_scene_img(self, scene_name, imgs_num = 10, random = 0):
    imgs_database = self.scenes_imgs_database[scene_name]
    # randomly select imgs from database (directory)
    if random:
      imgs = os.listdir(imgs_database)
      imgs = [os.path.join(imgs_database, img) for img in imgs]
      imgs = random.sample(imgs, imgs_num)
      return imgs
    else:
      imgs = os.listdir(imgs_database)
      imgs = [os.path.join(imgs_database, img) for img in imgs]
      return imgs[:imgs_num]
  
  def evaluate(self, ground_truth, estimated_graph:np.array):
    """
    evaluate the estimated graph with ground truth
    """
    self.get_TPR(run_times, ground_truth, estimated_graph)
    self.get_FPR(run_times, ground_truth, estimated_graph)
    self.get_SHD(ground_truth, estimated_graph)
    
  def get_TPR(self, gound_truth = None, estimated_graph = None):
    run_times = len(estimated_graph)
    ground_truth = ground_truth * run_times
    estimated_graph = np.sum(estimated_graph, axis=0)
    all_true_paths = np.sum(ground_truth)
    TPs = np.sum(estimated_graph[ground_truth>0])
    self.metrics["TPR"] = TPs / all_true_paths
    return TPs / all_true_paths
     
  def get_FPR(self, gound_truth = None, estimated_graph = None):
    run_times = len(estimated_graph)
    all_paths = sum(gound_truth.shape) * run_times
    ground_truth = ground_truth * run_times
    estimated_graph = np.sum(estimated_graph, axis=0)
    all_false_paths = all_paths - np.sum(ground_truth > 0)
    FPs = np.sum(estimated_graph[ground_truth == 0])
    self.metrics["FPR"] = FPs / all_false_paths
    return FPs / all_false_paths
  
  def get_SHD(self, gound_truth = None, estimated_graph = None):
    record = ""
    for each_estimated_graph in estimated_graph:
      temp = estimated_graph[estimated_graph != gound_truth]
      temp = np.sum(temp)
      if record == "":
        record = temp
      else:
        record += temp
        
    self.metrics["SHD"] = (record.item())/len(estimated_graph)
    return (record.item())/len(estimated_graph)
  
  
  
  
  def get_metrics(self):
    return self.metrics
    
if __name__ == "__main__":
  e = evaluation()
  print(len(e.get_scene_img("reflection")))
  scene = scene()
  scene = scene.get_scene("reflection")
  print(scene)
  
import pandas as pd
import math
import numpy as np
import os
import glob
import pickle
import glob
import time

from multiprocessing import Process # Intel Xeon Gold 6138 @2.00GHz has 20 cores and 40 threads

from helper_fcns import get_all_i_features

'''
Calculate features previously found to be related to mounting behavior from deeplabcut bodypart detections.
Change project_path, total_frames, and todo_list as needed 
'''

def run_me(total_frames, video_name, suffix, project_path):
  all_i_features = get_all_i_features(total_frames, video_name, suffix, project_path)

  # save all_i_features as pickle
  with open(project_path + "\\behaviors\\" + video_name + "_features.pickle", 'wb') as file:
      pickle.dump(all_i_features, file)

  return

if __name__ == '__main__':
  project_path = "D:\\example\\demo-me-2021-07-14\\"
  suffix = 'DLC_resnet50_demoJul14shuffle1_50000_assemblies'
  total_frames = 36000
  todo_list = ["221209_PZ70_1"]

  process_list = []
  for todo in todo_list:
    p = Process(target=run_me, args=(total_frames, todo, suffix, project_path,))
    process_list.append(p)
    p.start()
  
  for p in process_list:
    p.join()
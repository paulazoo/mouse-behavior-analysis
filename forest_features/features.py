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

def run_me(total_frames, video_name, h5_suffix, project_path):
  all_i_features = get_all_i_features(total_frames, video_name, h5_suffix, project_path)

  # save all_i_features as pickle
  with open(project_path + "\\behaviors\\" + video_name + "_features.pickle", 'wb') as file:
      pickle.dump(all_i_features, file)

  return

if __name__ == '__main__':
  project_path = "D:\\paulazhu\\demo-me-2021-07-14\\"
  #video_name = '221016_PZ70_1'
  h5_suffix = 'DLC_resnet50_demoJul14shuffle1_50000_el_filtered'
  #total_frames = 36000
  total_frames = 36000

  # takes a while
  print('running...')
  t = time.time()

  todo_list = ["221002_PZ71_1", \
                  "221009_PZ71_1", \
                  "221016_PZ70_1", \
                  "221016_PZ71_1", \
                  "221024_PZ70_1", \
                  "221024_PZ71_1", \
                  "221106_PZ70_1", \
                  "221106_PZ71_1", \
                  "221113_PZ87_1", \
                  "221113_PZ89_1", \
                  "221113_PZ90_1", \
                  "221119_PZ70_1", \
                  "221119_PZ71_1", \
                  "221120_PZ87_1", \
                  "221120_PZ89_1", \
                  "221120_PZ90_1", \
                  "221127_PZ87_1", \
                  "221127_PZ90_1", \
                  "221127_PZ70_1", \
                  "221127_PZ71_1", \
                    ]
                 
  #"221127_PZ89_1", \
  # "221009_PZ70_1", \ done?

  process_list = []
  for todo in todo_list:
    p = Process(target=run_me, args=(total_frames, todo, h5_suffix, project_path,))
    process_list.append(p)
    p.start()
  
  for p in process_list:
    p.join()

  elapsed = time.time() - t
  print(elapsed)
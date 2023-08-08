import os
import deeplabcut
import numpy as np
import pandas as pd
import pickle
import random
from pathlib import Path
import glob
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
from deeplabcut.utils.auxiliaryfunctions import edit_config

'''
Create deeplabcut videos with all detections
Change project_path, and todo_list as needed

if want to switch from running on Colab to running locally, need to copy over:
- video folder analyzed pickles
  - must go into video folder and select all and then download zip. Otherwise will miss files.
  - will give multiple zips probably
  - double check that you have the correct number of files downloaded
- dlc-models folder
- config.yaml
'''

project_path = "D:\\example\\demo-me-2021-07-14\\"
config_path = os.path.join(project_path, "config.yaml")
todo_list = ["221009_PZ74_2"]

SHUFFLE = 1
TRACK_METHOD = "ellipse"  # Could also be "box", but "ellipse" was found to be more robust on this dataset.
SNAPSHOT = 4 # not the actual number, but its index within the folder starting from 0
edit_config(config_path, {'snapshotindex': SNAPSHOT})

for video_name in todo_list:
  video = project_path + "videos\\"+video_name+".avi"
  deeplabcut.convert_detections2tracklets(
      config_path,
      [video],
      videotype='avi',
      shuffle=SHUFFLE,
      track_method=TRACK_METHOD,
      ignore_bodyparts=["tail1", "tail2", "tailend"],  # Some body parts can optionally be ignored during tracking for better assembly (but they are used later),
  )

# these are not actually necessary for just getting features, but are necessary for making deeplabcut final detection videos
for video_name in todo_list:
  video = project_path + "videos\\"+video_name+".avi"
  deeplabcut.stitch_tracklets(
    config_path,
    [video],
    videotype='avi',
    shuffle=SHUFFLE,
    track_method=TRACK_METHOD,
    n_tracks=2, # if 2 doesn't work, try 3 or 1 depending on how clear 2 mice is
  )
  #Filter the predictions to remove small jitter, if desired:
  deeplabcut.filterpredictions(config_path,
                                [video],
                                shuffle=SHUFFLE,
                                videotype='avi',
                                track_method = TRACK_METHOD)
import os
import deeplabcut
import numpy as np
import pandas as pd
import pickle
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
import random
from pathlib import Path
import glob

do_videos = 0

project_path = "D:\\paulazhu\\demo-me-2021-07-14\\"
config_path = os.path.join(project_path, "config.yaml")

SHUFFLE = 1
TRACK_METHOD = "ellipse"  # Could also be "box", but "ellipse" was found to be more robust on this dataset.

from deeplabcut.utils.auxiliaryfunctions import edit_config
SNAPSHOT = 4 # not the actual number, but its index within the folder starting from 0
edit_config(config_path, {'snapshotindex': SNAPSHOT})


if do_videos == 1:
    for video in glob.glob(project_path + "videos\\" + "*.avi"):
        deeplabcut.create_video_with_all_detections(
            config_path,
            [video],
            videotype='avi',
            shuffle=SHUFFLE,
            )

for video in [project_path + "videos\\221127_PZ90_1.avi"]:
# for video in glob.glob(project_path + "videos\\" + "*.avi"):
  deeplabcut.convert_detections2tracklets(
      config_path,
      [video],
      videotype='avi',
      shuffle=SHUFFLE,
      track_method=TRACK_METHOD,
      ignore_bodyparts=["tail1", "tail2", "tailend"],  # Some body parts can optionally be ignored during tracking for better assembly (but they are used later),
  )


# for video in glob.glob(project_path + "videos\\" + "*.avi"):
for video in [project_path + "videos\\221127_PZ90_1.avi"]:
  deeplabcut.stitch_tracklets(
    config_path,
    [video],
    videotype='avi',
    shuffle=SHUFFLE,
    track_method=TRACK_METHOD,
    n_tracks=2, # if 2 doesn't work, try 3 or 1 depending on how clear 2 mice is; also seems to mess up create_labeled_video when not 3
  )
  #Filter the predictions to remove small jitter, if desired:
  deeplabcut.filterpredictions(config_path,
                                [video],
                                shuffle=SHUFFLE,
                                videotype='avi',
                                track_method = TRACK_METHOD)
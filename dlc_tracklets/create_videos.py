import deeplabcut
import numpy as np
import pandas as pd
import pickle
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
from deeplabcut.utils.auxiliaryfunctions import edit_config
import random
from pathlib import Path
import glob
import os

from multiprocessing import Process # Intel Xeon Gold 6138 @2.00GHz has 20 cores and 40 threads


def run_me(video, config_path, SHUFFLE):
    deeplabcut.create_video_with_all_detections(
        config_path,
        [video],
        videotype='avi',
        shuffle=SHUFFLE,
        )
    return


if __name__ == '__main__':

    project_path = "D:\\paulazhu\\demo-me-2021-07-14\\"
    config_path = os.path.join(project_path, "config.yaml")

    SHUFFLE = 1
    TRACK_METHOD = "ellipse"  # Could also be "box", but "ellipse" was found to be more robust on this dataset.

    SNAPSHOT = 4 # not the actual number, but its index within the folder starting from 0
    edit_config(config_path, {'snapshotindex': SNAPSHOT})

    todo_list = ["221209_PZ70_1", \
                    "221209_PZ71_1", \
                    "221209_PZ87_1", \
                    "221209_PZ88_1", \
                    "221209_PZ89_1", \
                    "221209_PZ90_1"]

    process_list = []
    for todo in todo_list:
        video = project_path + "videos\\" + todo + ".avi"
        p = Process(target=run_me, args=(video, config_path, SHUFFLE,))
        process_list.append(p)
        p.start()
  
    for p in process_list:
        p.join()

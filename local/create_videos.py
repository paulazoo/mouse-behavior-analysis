import deeplabcut
import numpy as np
import pandas as pd
import pickle
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
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

    from deeplabcut.utils.auxiliaryfunctions import edit_config
    SNAPSHOT = 4 # not the actual number, but its index within the folder starting from 0
    edit_config(config_path, {'snapshotindex': SNAPSHOT})


    todo_list = ["221002_PZ71_1", \
                    "221127_PZ89_1", \
                    "221009_PZ70_1", \
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
                    "221009_PZ70_1", \
                    "221127_PZ89"
                    ]

    process_list = []
    for todo in todo_list:
        video = project_path + "videos\\" + todo + ".avi"
        p = Process(target=run_me, args=(video, config_path, SHUFFLE,))
        process_list.append(p)
        p.start()
  
    for p in process_list:
        p.join()

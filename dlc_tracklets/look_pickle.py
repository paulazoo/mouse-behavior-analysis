import pandas as pd
import math
import numpy as np
import os
import glob
import pickle
import glob
import time
 
 
project_path = "D:\\paulazhu\\demo-me-2021-07-14\\"
#video_name = '221016_PZ70_1'
h5_suffix = 'DLC_resnet50_demoJul14shuffle1_50000_el'
video_name = "221127_PZ90_1"

i= 1012

file = open(project_path+"videos\\221127_PZ90_1DLC_resnet50_demoJul14shuffle1_50000_assemblies.pickle", 'rb')

# dump information to that file
data = pickle.load(file)


# for assemblies pickle
i_data = data[i]
mouse1 = i_data[0]
mouse2 = i_data[1]


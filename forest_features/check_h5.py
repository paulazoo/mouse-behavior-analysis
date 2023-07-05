import pandas as pd
import math
import numpy as np
import os
import glob
import pickle
import glob
import time
 
from helper_fcns import get_data, get_feature_points

project_path = "D:\\paulazhu\\demo-me-2021-07-14\\"
#video_name = '221016_PZ70_1'
h5_suffix = 'DLC_resnet50_demoJul14shuffle1_50000_el'
video_name = "221127_PZ90_1"

i= 1012

print(project_path +'\\videos\\'+ video_name + h5_suffix + '.h5')
h5_file = pd.read_hdf(project_path +'\\videos\\'+ video_name + h5_suffix + '.h5')

mouse1_feature_points, mouse2_feature_points, relevant_area = get_feature_points(h5_file)

print(mouse1_feature_points['spine1'].loc[[i]])
print(mouse2_feature_points['spine1'].loc[[i]])
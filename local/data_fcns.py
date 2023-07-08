import pandas as pd
import math
import numpy as np
import os
import pickle
import glob
import scipy.io
import time

# HELPER FUNCTIONS

def get_data(individual, bodypart, data):
  # for assemblies pickle
  num_frames = 36000
  output = np.empty((num_frames,4))
  output[:] = np.nan
  for i in data:
      if i < 36000 and not (len(data[i]) < 2 and individual == 1): # individual==1 bc indexing from 0
          output[i, :]= data[i][individual][bodypart, :]
  return output


def get_feature_pts(data):
    # get all data ready for accessing as needed later on
    # getting m1 feature points
    m1_feature_pts = {}
    m1_feature_pts['snout'] = get_data(0, 0, data)
    m1_feature_pts['leftear'] = get_data(0, 1, data)
    m1_feature_pts['rightear'] = get_data(0, 2, data)
    m1_feature_pts['shoulder'] = get_data(0, 3, data)
    m1_feature_pts['spine1'] = get_data(0, 4, data)
    m1_feature_pts['spine2'] = get_data(0, 5, data)
    m1_feature_pts['spine3'] = get_data(0, 6, data)
    m1_feature_pts['spine4'] = get_data(0, 7, data)
    m1_feature_pts['tailbase'] = get_data(0, 8, data)
    # getting m2 feature points
    m2_feature_pts = {}
    m2_feature_pts['snout'] = get_data(1, 0, data)
    m2_feature_pts['leftear'] = get_data(1, 1,  data)
    m2_feature_pts['rightear'] = get_data(1, 2,  data)
    m2_feature_pts['shoulder'] = get_data(1, 3,  data)
    m2_feature_pts['spine1'] = get_data(1, 4,  data)
    m2_feature_pts['spine2'] = get_data(1, 5,  data)
    m2_feature_pts['spine3'] = get_data(1, 6,  data)
    m2_feature_pts['spine4'] = get_data(1, 7,  data)
    m2_feature_pts['tailbase'] = get_data(1, 8,  data)
    # male_side_vec = [x, y, width, height]; female_side_vec = [x, y, width, height]
    relevant_area = [0, 0, 1000, 1000]
    return m1_feature_pts, m2_feature_pts, relevant_area

import pandas as pd
import math
import numpy as np
import os
import pickle
import glob
import scipy.io
import time

from data_fcns import get_feature_pts
from euclidean_fcns import euclid_dist, euclid_angle
from mice_exist_fcns import check_mice_exist, check_female_side_mice

# REST OF FEATURES HERE-----------------------------------------------------------------------
def get_torso_dists(i, first_m_feature_pts, second_m_feature_pts, torso_bodyparts):
  dists = np.zeros((7,7))

  for bodypart_i in range(0, len(torso_bodyparts)):
    for bodypart_j in range(0, len(torso_bodyparts)):
      dists[bodypart_i, bodypart_j] = euclid_dist(first_m_feature_pts[torso_bodyparts[bodypart_i]][i, :], \
                                                  second_m_feature_pts[torso_bodyparts[bodypart_j]][i, :])
  return dists

def long_short(i, first_m_feature_pts, second_m_feature_pts, torso_bodyparts):
  first_m_dists = np.zeros((7,7))
  second_m_dists = np.zeros((7,7))
  for bodypart_i in range(0, len(torso_bodyparts)):
    for bodypart_j in range(0, len(torso_bodyparts)):
      # euclid_dist should be nan if any one of the input bodyparts is not present
      first_m_dists[bodypart_i, bodypart_j] = euclid_dist(first_m_feature_pts[torso_bodyparts[bodypart_i]][i, :], first_m_feature_pts[torso_bodyparts[bodypart_j]][i, :])
      second_m_dists[bodypart_i, bodypart_j] = euclid_dist(second_m_feature_pts[torso_bodyparts[bodypart_i]][i, :], second_m_feature_pts[torso_bodyparts[bodypart_j]][i, :])

  # nanmax should be 0 if there was only 1 bodypart
  if np.nanmax(first_m_dists) > np.nanmax(second_m_dists):
    long_m_feature_pts = first_m_feature_pts
    short_m_feature_pts = second_m_feature_pts
    long_m_dists = first_m_dists
    short_m_dists = second_m_dists
  else:
    long_m_feature_pts = second_m_feature_pts
    short_m_feature_pts = first_m_feature_pts
    long_m_dists = second_m_dists
    short_m_dists = first_m_dists

  return long_m_feature_pts, short_m_feature_pts, long_m_dists, short_m_dists

# GET FEATURES----------------------------------------------------------------------------
def get_i_features(i, m1_feature_pts, m2_feature_pts, relevant_area, female_side_vec):

  first_m_feature_pts, second_m_feature_pts = check_mice_exist(i, m1_feature_pts, m2_feature_pts,relevant_area)
  
  if second_m_feature_pts == 0:
    num_features = 9
    return np.zeros((1, num_features))

  else:
    # check female side
    female_side_mice = check_female_side_mice(i, m1_feature_pts, \
                                              m2_feature_pts, \
                                                female_side_vec)

    # torso dist stuff
    torso_bodyparts = ['snout', 'shoulder', 'spine1', 'spine2', 'spine3', 'spine4', 'tailbase']
    torso_dists = get_torso_dists(i, first_m_feature_pts, second_m_feature_pts, torso_bodyparts)
    first_min_id,second_min_id = np.unravel_index(np.nanargmin(torso_dists), torso_dists.shape)
    min_dist = torso_dists[first_min_id,second_min_id]
    min_posteriority_diff = abs(first_min_id - second_min_id)
    first_max_id,second_max_id = np.unravel_index(np.nanargmax(torso_dists), torso_dists.shape)
    max_dist = torso_dists[first_max_id,second_max_id]
    max_posteriority_diff = abs(first_max_id - second_max_id)

    # angle stuff
    long_m_feature_pts, short_m_feature_pts, long_m_dists, short_m_dists = long_short(i, first_m_feature_pts, second_m_feature_pts, torso_bodyparts)
    long_ids = np.array([np.unravel_index(np.nanargmax(long_m_dists), long_m_dists.shape)])
    short_ids = np.array([np.unravel_index(np.nanargmax(short_m_dists), short_m_dists.shape)])
    long_posterior = torso_bodyparts[np.max(long_ids)]
    long_anterior = torso_bodyparts[np.min(long_ids)]
    short_posterior = torso_bodyparts[np.max(short_ids)]
    short_anterior = torso_bodyparts[np.min(short_ids)]
    # more angle stuff
    posterior_ang = euclid_angle(long_m_feature_pts[long_anterior][i, :], long_m_feature_pts[long_posterior][i, :], short_m_feature_pts[short_posterior][i, :])
    anterior_ang = euclid_angle(long_m_feature_pts[long_posterior][i, :], long_m_feature_pts[long_anterior][i, :], short_m_feature_pts[short_anterior][i, :])
    
    # mouse line angle stuff
    long_m_line = long_m_feature_pts[long_anterior][i, :]-long_m_feature_pts[long_posterior][i, :]
    short_m_line = short_m_feature_pts[short_anterior][i, :]-short_m_feature_pts[short_posterior][i, :]
    line_ang = euclid_angle(long_m_line, [0,0, 0,0], short_m_line)

    return np.array([1, female_side_mice, \
                     min_dist,min_posteriority_diff,max_dist,max_posteriority_diff, \
                      posterior_ang, anterior_ang, line_ang])



def get_all_i_features(total_frames, video_name, suffix, project_path):
  print(project_path +'\\videos\\'+ video_name + suffix + '.pickle')
  file = open(project_path +'\\videos\\'+ video_name + suffix + '.pickle', 'rb')
  data = pickle.load(file)

  # area_vec = [area_startx, area_starty, area_distx, area_disty]
  female_side_mat = scipy.io.loadmat(project_path + "\\behaviors\\" + video_name + "_female_side.mat")
  female_side_vec = female_side_mat['croprect'][0]

  m1_feature_pts, m2_feature_pts, relevant_area = get_feature_pts(data)
  
  # now go find features for each frame
  num_features = 9
  all_i_features = np.empty([total_frames, num_features])
  for frame in range(total_frames):
      all_i_features[frame, 0:num_features] = get_i_features(frame, \
                                                          m1_feature_pts, m2_feature_pts, \
                                                          relevant_area, female_side_vec)
      
      if np.any(np.isnan(all_i_features[frame, 0:num_features])):
        print('nans detected')
        print(all_i_features[frame, 0:num_features])

  return all_i_features

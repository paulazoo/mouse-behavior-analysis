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

# EUCLID STUFF HERE----------------------------------------------------------------
def within_area(area_vector, input_coor):
  area_startx = area_vector[0]
  area_starty = area_vector[1]
  area_distx = area_vector[2]
  area_disty = area_vector[3]
  x = input_coor[0]
  y = input_coor[1]
  if (area_startx <= x <= (area_startx+area_distx)) and (area_starty <= y <= (area_starty+area_disty)):
    result = 1
  else:
    result = 0
  return result

# euclid_dist(first_m_feature_pts['snout'].loc[[100]], second_m_feature_pts['shoulder'].loc[[100]]])
def euclid_dist(point1_coor, point2_coor):
  point1 = np.array((point1_coor[0], point1_coor[1]))
  point2 = np.array((point2_coor[0], point2_coor[1]))
  output_dist = np.linalg.norm(point1 - point2)
  # but if any coordinate is NaN, then output_dist is 0
  if np.isnan(point1).any() or np.isnan(point2).any():
    output_dist = 0
  return output_dist

def euclid_angle(pointa_coor, pointb_coor, pointc_coor):
  #angle_{pointa, pointb, pointc}
  # euclid_angle(first_m_feature_pts['snout'].loc[[100]], first_m_feature_pts['shoulder'].loc[[100]]], second_m_feature_pts['snout'].loc[[100]]])
  a = np.array((pointa_coor[0], pointa_coor[1]))
  b = np.array((pointb_coor[0], pointb_coor[1]))
  c = np.array((pointc_coor[0], pointc_coor[1]))

  ba = a - b
  bc = c - b

  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  angle = np.arccos(cosine_angle)

  output_ang = np.degrees(angle)

  # but if any coordinate is NaN, then output_ang is 0
  if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
    output_ang = 0

  return output_ang

# BODYPARTS IN AREAS HERE---------------------------------------------------------------
def torso_in_area(area_vector, feature_pts, i):
  # finds if any part of torso at all is in area

  if within_area(area_vector, feature_pts['snout'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['shoulder'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine1'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine2'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine3'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine4'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['tailbase'][i, :]):
    return 1
  else:
    return 0

def midtorso_in_area(area_vector, feature_pts, i):
  # finds if any part of torso at all is in area

  if within_area(area_vector, feature_pts['spine1'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine2'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine3'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine4'][i, :]):
    return 1
  else:
    return 0

def ear_or_torso_in_area(area_vector, feature_pts, i):
  # finds if any part of torso or ear at all is in area
  if within_area(area_vector, feature_pts['snout'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['shoulder'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine1'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine2'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine3'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['spine4'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['tailbase'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['leftear'][i, :]):
    return 1
  elif within_area(area_vector, feature_pts['rightear'][i, :]):
    return 1
  else:
    return 0

def snout_in_area(area_vector, feature_pts, i):
  # finds if any part of snout at all is in area

  if within_area(area_vector, feature_pts['snout'][i, :]):
    result = 1
  else:
    result = 0
  return result

def shoulder_in_area(area_vector, feature_pts, i):
  # finds if any part of shoulder at all is in area

  if within_area(area_vector, feature_pts['shoulder'][i, :]):
    result = 1
  else:
    result = 0
  return result

# CHECK MICE HERE---------------------------------------------------------------------
def check_mice_exist(i, m1_feature_pts, m2_feature_pts, relevant_area):
  # 2 mice detected
  if ear_or_torso_in_area(relevant_area, m1_feature_pts, i) and \
    ear_or_torso_in_area(relevant_area, m2_feature_pts, i):
    first_m_feature_pts = m1_feature_pts
    second_m_feature_pts = m2_feature_pts
  # 1 m detected
  elif ear_or_torso_in_area(relevant_area, m1_feature_pts, i):
    first_m_feature_pts = m1_feature_pts
    second_m_feature_pts = 0
  elif ear_or_torso_in_area(relevant_area, m2_feature_pts, i):
    first_m_feature_pts = m2_feature_pts
    second_m_feature_pts = 0
  # no m detected
  else:
    first_m_feature_pts = 0
    second_m_feature_pts = 0
  return first_m_feature_pts, second_m_feature_pts

def check_female_side_mice(i, m1_feature_pts, m2_feature_pts, female_side_vec):
  if midtorso_in_area(female_side_vec, m1_feature_pts, i) and \
    midtorso_in_area(female_side_vec, m2_feature_pts, i):
    return 1
  else:
    return 0

# REST OF FEATURES HERE-----------------------------------------------------------------------
def get_torso_dists(i, first_m_feature_pts, second_m_feature_pts):
  dists = np.zeros((7,7))

  torso_bodyparts = ['snout', 'shoulder', 'spine1', 'spine2', 'spine3', 'spine4', 'tailbase']

  for bodypart_i in range(0, len(torso_bodyparts)):
    for bodypart_j in range(0, len(torso_bodyparts)):
      dists[bodypart_i, bodypart_j] = euclid_dist(first_m_feature_pts[torso_bodyparts[bodypart_i]][i, :], \
                                                  second_m_feature_pts[torso_bodyparts[bodypart_j]][i, :])
  return dists

def long_short(i, first_m_feature_pts, second_m_feature_pts):
  torso_bodyparts = ['snout', 'shoulder', 'spine1', 'spine2', 'spine3', 'spine4', 'tailbase']

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
    # num_features = 8
    return np.array([0, 0, 0,0,0,0, 0,0])

  else:
    # check female side
    female_side_mice = check_female_side_mice(i, m1_feature_pts, \
                                              m2_feature_pts, \
                                                female_side_vec)

    # torso dist stuff
    torso_dists = get_torso_dists(i, first_m_feature_pts, second_m_feature_pts)
    first_min_id,second_min_id = np.unravel_index(np.nanargmin(torso_dists), torso_dists.shape)
    min_dist = torso_dists[first_min_id,second_min_id]
    min_posteriority_diff = abs(first_min_id - second_min_id)
    first_max_id,second_max_id = np.unravel_index(np.nanargmax(torso_dists), torso_dists.shape)
    max_dist = torso_dists[first_max_id,second_max_id]
    max_posteriority_diff = abs(first_max_id - second_max_id)

    # angle stuff
    long_m_feature_pts, short_m_feature_pts, long_m_dists, short_m_dists = long_short(i, first_m_feature_pts, second_m_feature_pts)
    long_ids = np.array([np.unravel_index(np.nanargmax(long_m_dists), long_m_dists.shape)])
    short_ids = np.array([np.unravel_index(np.nanargmax(short_m_dists), short_m_dists.shape)])
    torso_bodyparts = ['snout', 'shoulder', 'spine1', 'spine2', 'spine3', 'spine4', 'tailbase']
    long_posterior = torso_bodyparts[np.max(long_ids)]
    long_anterior = torso_bodyparts[np.min(long_ids)]
    short_posterior = torso_bodyparts[np.max(short_ids)]
    short_anterior = torso_bodyparts[np.min(short_ids)]
    # more angle stuff
    posterior_ang = euclid_angle(long_m_feature_pts[long_anterior][i, :], long_m_feature_pts[long_posterior][i, :], short_m_feature_pts[short_posterior][i, :])
    anterior_ang = euclid_angle(long_m_feature_pts[long_posterior][i, :], long_m_feature_pts[long_anterior][i, :], short_m_feature_pts[short_anterior][i, :])

    return np.array([1, female_side_mice, \
                     min_dist,min_posteriority_diff,max_dist,max_posteriority_diff, \
                      posterior_ang, anterior_ang])



def get_all_i_features(total_frames, video_name, suffix, project_path):
  print(project_path +'\\videos\\'+ video_name + suffix + '.pickle')
  file = open(project_path +'\\videos\\'+ video_name + suffix + '.pickle', 'rb')
  data = pickle.load(file)

  # area_vec = [area_startx, area_starty, area_distx, area_disty]
  female_side_mat = scipy.io.loadmat(project_path + "\\behaviors\\" + video_name + "_female_side.mat")
  female_side_vec = female_side_mat['croprect'][0]
  male_side_mat = scipy.io.loadmat(project_path + "\\behaviors\\" + video_name + "_male_side.mat")
  male_side_vec = male_side_mat['croprect'][0]

  m1_feature_pts, m2_feature_pts, relevant_area = get_feature_pts(data)
  
  # now go find features for each frame
  num_features = 8
  all_i_features = np.empty([total_frames, num_features])
  for frame in range(total_frames):
      all_i_features[frame, 0:num_features] = get_i_features(frame, \
                                                          m1_feature_pts, m2_feature_pts, \
                                                          relevant_area, female_side_vec)
  return all_i_features

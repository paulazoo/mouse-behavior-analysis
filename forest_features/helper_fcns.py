import pandas as pd
import math
import numpy as np
import os
import pickle
import glob
import scipy.io
import time

# HELPER FUNCTIONS

def get_data(individual, bodypart, h5_file):
  mouse_data = h5_file.xs(individual,level='individuals',axis=1)
  out_data = mouse_data.xs(bodypart,level='bodyparts',axis=1)
  out_data.columns = out_data.columns.droplevel("scorer")
  out_data_copy = out_data.copy()
  output = out_data.copy()
  # if missing a lot of body parts:
  # for i in range(out_data_copy) go by every 5:
  #  between i:i+4 find index value with highest "likelihood"
  #  output[i:i+4] = x and y from max likelihood
  return output


# get all data ready for accessing as needed later on

# names of csv individuals
individual1 = 'ind1'
individual2 = 'ind2'
#individual3 = 'ind3'

def get_feature_points(h5_file, female_side_vec, male_side_vec):
    # getting feature points
    mouse1_feature_points = {}
    mouse1_feature_points['snout'] = get_data('ind1', 'snout', h5_file)
    mouse1_feature_points['leftear'] = get_data('ind1', 'leftear', h5_file)
    mouse1_feature_points['rightear'] = get_data('ind1', 'rightear', h5_file)
    mouse1_feature_points['shoulder'] = get_data('ind1', 'shoulder', h5_file)
    mouse1_feature_points['spine1'] = get_data('ind1', 'spine1', h5_file)
    mouse1_feature_points['spine2'] = get_data('ind1', 'spine2', h5_file)
    mouse1_feature_points['spine3'] = get_data('ind1', 'spine3', h5_file)
    mouse1_feature_points['spine4'] = get_data('ind1', 'spine4', h5_file)
    mouse1_feature_points['tailbase'] = get_data('ind1', 'tailbase', h5_file)

    mouse2_feature_points = {}
    mouse2_feature_points['snout'] = get_data('ind2', 'snout', h5_file)
    mouse2_feature_points['leftear'] = get_data('ind1', 'leftear', h5_file)
    mouse2_feature_points['rightear'] = get_data('ind1', 'rightear', h5_file)
    mouse2_feature_points['shoulder'] = get_data('ind2', 'shoulder', h5_file)
    mouse2_feature_points['spine1'] = get_data('ind2', 'spine1', h5_file)
    mouse2_feature_points['spine2'] = get_data('ind2', 'spine2', h5_file)
    mouse2_feature_points['spine3'] = get_data('ind2', 'spine3', h5_file)
    mouse2_feature_points['spine4'] = get_data('ind2', 'spine4', h5_file)
    mouse2_feature_points['tailbase'] = get_data('ind2', 'tailbase', h5_file)

    # commented out because ran with n_tracks = 2
    #mouse3_feature_points = {}
    #mouse3_feature_points['snout'] = get_data('mus3', 'snout', h5_file)
    #mouse3_feature_points['shoulder'] = get_data('mus3', 'shoulder', h5_file)
    #mouse3_feature_points['spine1'] = get_data('mus3', 'spine1', h5_file)
    #mouse3_feature_points['spine2'] = get_data('mus3', 'spine2', h5_file)
    #mouse3_feature_points['spine3'] = get_data('mus3', 'spine3', h5_file)
    #mouse3_feature_points['spine4'] = get_data('mus3', 'spine4', h5_file)
    #mouse3_feature_points['tailbase'] = get_data('mus3', 'tailbase', h5_file)

    # male_side_vec = [x, y, width, height]; female_side_vec = [x, y, width, height]
    relevant_area = female_side_vec.copy()
    # relevant area is female_side_vec + male_side_vec size exactly right next to each other with 50 buffer
    relevant_area[2] = female_side_vec[2] + male_side_vec[2] + 50
    return mouse1_feature_points, mouse2_feature_points, relevant_area

def within_area(area_vector, input_coor):
  area_startx = area_vector[0]
  area_starty = area_vector[1]
  area_distx = area_vector[2]
  area_disty = area_vector[3]
  x = input_coor["x"].iloc[0]
  y = input_coor["y"].iloc[0]
  if (area_startx <= x <= (area_startx+area_distx)) and (area_starty <= y <= (area_starty+area_disty)):
    result = 1
  else:
    result = 0
  return result

# euclid_dist(first_mouse_feature_points['snout'].loc[[100]], second_mouse_feature_points['shoulder'].loc[[100]]])
def euclid_dist(point1_coor, point2_coor):
  point1 = np.array((point1_coor["x"].iloc[0], point1_coor["y"].iloc[0]))
  point2 = np.array((point2_coor["x"].iloc[0], point2_coor["y"].iloc[0]))
  output_dist = np.linalg.norm(point1 - point2)
  # but if any coordinate is NaN, then output_dist is 0
  if np.isnan(point1).any() or np.isnan(point2).any():
    output_dist = 0
  return output_dist

def euclid_angle(pointa_coor, pointb_coor, pointc_coor):
  #angle_{pointa, pointb, pointc}
  # euclid_angle(first_mouse_feature_points['snout'].loc[[100]], first_mouse_feature_points['shoulder'].loc[[100]]], second_mouse_feature_points['snout'].loc[[100]]])
  a = np.array((pointa_coor["x"].iloc[0], pointa_coor["y"].iloc[0]))
  b = np.array((pointb_coor["x"].iloc[0], pointb_coor["y"].iloc[0]))
  c = np.array((pointc_coor["x"].iloc[0], pointc_coor["y"].iloc[0]))

  ba = a - b
  bc = c - b

  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  angle = np.arccos(cosine_angle)

  output_ang = np.degrees(angle)

  # but if any coordinate is NaN, then output_ang is 0
  if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
    output_ang = 0

  return output_ang

def torso_in_area(area_vector, feature_points, i):
  # finds if any part of torso at all is in area

  if within_area(area_vector, feature_points['snout'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['shoulder'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['spine1'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['spine2'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['spine3'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['spine4'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['tailbase'].loc[[i]]):
    return 1
  else:
    return 0
  
def ear_or_torso_in_area(area_vector, feature_points, i):
  # finds if any part of torso at all is in area

  if within_area(area_vector, feature_points['snout'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['shoulder'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['spine1'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['spine2'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['spine3'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['spine4'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['tailbase'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['leftear'].loc[[i]]):
    return 1
  elif within_area(area_vector, feature_points['rightear'].loc[[i]]):
    return 1
  else:
    return 0

def snout_in_area(area_vector, feature_points, i):
  # finds if any part of snout at all is in area

  if within_area(area_vector, feature_points['snout'].loc[[i]]):
    result = 1
  else:
    result = 0
  return result

def shoulder_in_area(area_vector, feature_points, i):
  # finds if any part of shoulder at all is in area

  if within_area(area_vector, feature_points['shoulder'].loc[[i]]):
    result = 1
  else:
    result = 0
  return result


def check_mice_exist(i, mouse1_feature_points, mouse2_feature_points, relevant_area):
  # 2 mice detected
  if ear_or_torso_in_area(relevant_area, mouse1_feature_points, i) and \
    ear_or_torso_in_area(relevant_area, mouse2_feature_points, i):
    first_mouse_feature_points = mouse1_feature_points
    second_mouse_feature_points = mouse2_feature_points

  #elif torso_in_area(relevant_area,mouse1_feature_points, i) and torso_in_area(relevant_area, 3, i):
    # first_mouse_feature_points = mouse1_feature_points
    # second_mouse_feature_points = mouse3_feature_points

  #elif torso_in_area(relevant_area, mouse2_feature_points, i) and torso_in_area(relevant_area, 3, i):
  #  first_mouse_feature_points = mouse2_feature_points
  #  second_mouse_feature_points = mouse3_feature_points


  # 1 mouse detected
  elif ear_or_torso_in_area(relevant_area, mouse1_feature_points, i):
    first_mouse_feature_points = mouse1_feature_points
    second_mouse_feature_points = 0

  elif ear_or_torso_in_area(relevant_area, mouse2_feature_points, i):
    first_mouse_feature_points = mouse2_feature_points
    second_mouse_feature_points = 0

  #elif torso_in_area(relevant_area, 3, i):
  #  first_mouse_feature_points = mouse3_feature_points
  #  second_mouse_feature_points = 0

  # no mouse detected
  else:
    first_mouse_feature_points = 0
    second_mouse_feature_points = 0

  return first_mouse_feature_points, second_mouse_feature_points

def check_female_side_mice(i, mouse1_feature_points, mouse2_feature_points, female_side_vec):
  if ear_or_torso_in_area(female_side_vec, mouse1_feature_points, i) and \
    ear_or_torso_in_area(female_side_vec, mouse2_feature_points, i):
    return 1
  else:
    return 0
  
def get_torso_dists(i, first_mouse_feature_points, second_mouse_feature_points):
  dists = np.zeros((7,7))

  torso_bodyparts = ['snout', 'shoulder', 'spine1', 'spine2', 'spine3', 'spine4', 'tailbase']

  for bodypart_i in range(0, len(torso_bodyparts)):
    for bodypart_j in range(0, len(torso_bodyparts)):
      dists[bodypart_i, bodypart_j] = euclid_dist(first_mouse_feature_points[torso_bodyparts[bodypart_i]].loc[[i]], second_mouse_feature_points[torso_bodyparts[bodypart_j]].loc[[i]])
  return dists

def long_short(i, first_mouse_feature_points, second_mouse_feature_points):
  torso_bodyparts = ['snout', 'shoulder', 'spine1', 'spine2', 'spine3', 'spine4', 'tailbase']

  first_mouse_dists = np.zeros((7,7))
  second_mouse_dists = np.zeros((7,7))
  for bodypart_i in range(0, len(torso_bodyparts)):
    for bodypart_j in range(0, len(torso_bodyparts)):
      # euclid_dist should be nan if any one of the input bodyparts is not present
      first_mouse_dists[bodypart_i, bodypart_j] = euclid_dist(first_mouse_feature_points[torso_bodyparts[bodypart_i]].loc[[i]], first_mouse_feature_points[torso_bodyparts[bodypart_j]].loc[[i]])
      second_mouse_dists[bodypart_i, bodypart_j] = euclid_dist(second_mouse_feature_points[torso_bodyparts[bodypart_i]].loc[[i]], second_mouse_feature_points[torso_bodyparts[bodypart_j]].loc[[i]])

  # nanmax should be 0 if there was only 1 bodypart
  if np.nanmax(first_mouse_dists) > np.nanmax(second_mouse_dists):
    long_mouse_feature_points = first_mouse_feature_points
    short_mouse_feature_points = second_mouse_feature_points
    long_mouse_dists = first_mouse_dists
    short_mouse_dists = second_mouse_dists
  else:
    long_mouse_feature_points = second_mouse_feature_points
    short_mouse_feature_points = first_mouse_feature_points
    long_mouse_dists = second_mouse_dists
    short_mouse_dists = first_mouse_dists

  return long_mouse_feature_points, short_mouse_feature_points, long_mouse_dists, short_mouse_dists

def get_i_features(i, mouse1_feature_points, mouse2_feature_points, relevant_area, female_side_vec):

  first_mouse_feature_points, second_mouse_feature_points = check_mice_exist(i, mouse1_feature_points, \
                                                                             mouse2_feature_points, \
                                                                                relevant_area)
  if second_mouse_feature_points == 0:
    return np.array([0, 0, 0,0,0,0, 0,0])

  else:
    # check female side
    female_side_mice = check_female_side_mice(i, mouse1_feature_points, \
                                              mouse2_feature_points, \
                                                female_side_vec)

    # torso dist stuff
    torso_dists = get_torso_dists(i, first_mouse_feature_points, second_mouse_feature_points)
    first_min_id,second_min_id = np.unravel_index(np.nanargmin(torso_dists), torso_dists.shape)
    min_dist = torso_dists[first_min_id,second_min_id]
    min_posteriority_diff = abs(first_min_id - second_min_id)
    first_max_id,second_max_id = np.unravel_index(np.nanargmax(torso_dists), torso_dists.shape)
    max_dist = torso_dists[first_max_id,second_max_id]
    max_posteriority_diff = abs(first_max_id - second_max_id)

    # angle stuff
    long_mouse_feature_points, short_mouse_feature_points, long_mouse_dists, short_mouse_dists = long_short(i, first_mouse_feature_points, second_mouse_feature_points)
    long_ids = np.array([np.unravel_index(np.nanargmax(long_mouse_dists), long_mouse_dists.shape)])
    short_ids = np.array([np.unravel_index(np.nanargmax(short_mouse_dists), short_mouse_dists.shape)])
    torso_bodyparts = ['snout', 'shoulder', 'spine1', 'spine2', 'spine3', 'spine4', 'tailbase']
    long_posterior = torso_bodyparts[np.max(long_ids)]
    long_anterior = torso_bodyparts[np.min(long_ids)]
    short_posterior = torso_bodyparts[np.max(short_ids)]
    short_anterior = torso_bodyparts[np.min(short_ids)]
    # more angle stuff
    posterior_ang = euclid_angle(long_mouse_feature_points[long_anterior].loc[[i]], long_mouse_feature_points[long_posterior].loc[[i]], short_mouse_feature_points[short_posterior].loc[[i]])
    anterior_ang = euclid_angle(long_mouse_feature_points[long_posterior].loc[[i]], long_mouse_feature_points[long_anterior].loc[[i]], short_mouse_feature_points[short_anterior].loc[[i]])

    return np.array([1, female_side_mice, min_dist,min_posteriority_diff,max_dist,max_posteriority_diff, posterior_ang, anterior_ang])



def get_all_i_features(total_frames, video_name, h5_suffix, project_path):
  print(project_path +'\\videos\\'+ video_name + h5_suffix + '.h5')
  h5_file = pd.read_hdf(project_path +'\\videos\\'+ video_name + h5_suffix + '.h5')

  # area_vec = [area_startx, area_starty, area_distx, area_disty]
  female_side_mat = scipy.io.loadmat(project_path + "\\behaviors\\" + video_name + "_female_side.mat")
  female_side_vec = female_side_mat['croprect'][0]
  male_side_mat = scipy.io.loadmat(project_path + "\\behaviors\\" + video_name + "_male_side.mat")
  male_side_vec = male_side_mat['croprect'][0]

  mouse1_feature_points, mouse2_feature_points, relevant_area = get_feature_points(h5_file, \
                                                                                  female_side_vec, \
                                                                                    male_side_vec)
  
  # now go find features for each frame
  num_features = 8
  all_i_features = np.empty([total_frames, num_features])
  for frame in range(total_frames):
      all_i_features[frame, 0:num_features] = get_i_features(frame, \
                                                          mouse1_feature_points, mouse2_feature_points, \
                                                          relevant_area, female_side_vec)
  return all_i_features

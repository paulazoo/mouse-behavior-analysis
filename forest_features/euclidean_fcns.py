import pandas as pd
import math
import numpy as np

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
def euclid_dist(point1coor, point2coor):
  point1 = point1coor[0:2]
  point2 = point2coor[0:2]

  output_dist = np.linalg.norm(point1 - point2)
  # but if any coordinate is NaN, then output_dist is 0
  if np.isnan(point1).any() or np.isnan(point2).any():
    output_dist = 0
  return output_dist

def euclid_angle(acoor, bcoor, ccoor):
  a = acoor[0:2]
  b = bcoor[0:2]
  c = ccoor[0:2]

  # if any coordinate is NaN, then output_ang is 0
  if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
    output_ang = 0
  # if any coordinate is the same as another, then output_ang is 0 for ([0,0])
  elif np.array_equal(a,b) or np.array_equal(a,b) or np.array_equal(a,b):
    output_ang = 0
  else:
    # otherwise calculate angle
    #angle_{pointa, pointb, pointc} where pointa = [x, y]
    ba = a - b
    bc = c - b
    # avoid invalid value encountered in double_scalars
    # when bc is super small
    if np.linalg.norm(ba) < 1 or np.linalg.norm(bc) < 1:
      output_ang = 0
    else:
      cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
      angle = np.arccos(cosine_angle)

      output_ang = np.degrees(angle)

  return output_ang


# BODYPART IN AREA FUNCTIONS----------------------------------------
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
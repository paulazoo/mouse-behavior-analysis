import pandas as pd
import numpy as np

from euclidean_fcns import ear_or_torso_in_area, midtorso_in_area

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

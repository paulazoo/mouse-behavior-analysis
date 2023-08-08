# General:
Run through deeplabcut model --> Calculate features from bodyparts --> Run through classifier to find putative mounts
__bold = to run (from this directory)__
_slanted = timing_
`code format = a file or something in model directory`

# 1) Local Video Editing:
- __video_cropping/get_area.m__: _(1m)_ Select female-only side area of the cage, name as '[video_name]_female_side' in the gui. Outputs `[video_name]_female_side.mat` which is needed for one of the classifier features.
- Upload `[video_name]_female_side.mat` to `behaviors` folder in the model directory.

# 2) Cloud (GPU limited) Deeplabcut:
- Upload videos to `videos` folder in the model directory
- __./run_dlc_multianimal.ipynb__: _(several m)_ Outputs `.pickle` detection files to `videos` folder

# 3) Cloud or Local (CPU limited) Deeplabcut:
- If switching to run locally, will need to download detections, `config.yml`, model directory structure
- __dlc_tracklets/make_tracklets.py__: _(1m)_ Outputs more `.pickle` files and an `.h5` file to `videos` folder
- __dlc_tracklets/create_videos.py__: _(a few min)_ this is necessary for creating labeled behavior videos to check. Outputs `.mp4` detections video in `videos` folder

# 4) Calculate features (CPU limited):
- __forest_features/features.py__: _(1m)_ Outputs `all_i_features` containing all features as a `_features.pickle` in `behaviors` folder

# 5) Run classifier
- __./run_mount_forest.ipynb__: _(1m)_ Currently only individually classifies each frame independently, but then averages within each 1 second. Outputs a `_pred.xlsx` of mount start frames/durations and a `_rfc.mp4` labelled behavior video to `behaviors` folder.

# 6) Manually check mounts for intromission
- _(2m)_ Go to the start of each mount listed in `_pred.xlsx` file and check video manually for mount vs intromission
- Classifier is good at not missing any mounts, but may occasionally give false positives


# Troubleshooting:
- Try not to include human hands
- Try to crop video to exclude walls as much as possible, since plastic reflections of mice sometimes also confuse deeplabcut
- If a mouse manages to climb up onto wall ledge out of camera and/or jump and hang off edge, classifier will think it's a mount because will think 2nd mouse is under 1st mouse
- Matlab scripts are slow bc Matlab, but can open and run several Matlab instances in parallel
- some of the python scripts already have parallel coded in, so can edit depending on # of cores
- __./run_dlc_multianimal.ipynb__ is slow, but you can make several gmails and copies of this in Google Colab, then run all videos at same time
- __dlc_tracklets/make_tracklets.py__ I decided to take raw pose estimation pickle data prior to any stitching/filtering/p-value dropping for classifier features, so technically don't need to run anything past dlc's _convert_detections2tracklets_ step, but the rest of dlc's workflow is still needed if wanted to make dlc's full, filtered, stitched detections video


## Other notes:
__Mount vs Intromission__:
So far, unable to make a classifier to accurately detect mount vs intromission, but this might be possible if temporal data (n frame features ahead or before in time) could be included; however, this many features/dimensions would require a lot more training frames

__Workflow for other behaviors__:
Train from current deeplabcut snapshot
Run a new pca on different bodypart angles/distances/areas for mice to be in
Then make a new SMOTE or otherwise imbalance-adjusted (because 1/100 frames will be the behavior) rfc on the most predictive features.

__Amount of training data__:
In total, ~300 accumulated training frames to go from the open-field 3 mouse demo model (publicly available) to this most recent version
However, also worked fine back when only trained with 50 frames

__Constants used__:
- 36000 frames (20m videos)
- 30 fps
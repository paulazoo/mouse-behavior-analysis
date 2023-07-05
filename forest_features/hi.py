project_path = "D:\\paulazhu\\demo-me-2021-07-14\\"
v = project_path+"//videos//221127_PZ90_1.avi"
asse_p = project_path+"//videos//221127_PZ90_1DLC_resnet50_demoJul14shuffle1_50000_assemblies.pickle"
deeplabcut.utils.make_labeled_video.create_video_from_pickled_tracks(v, asse_p, destfolder=project_path, output_name="hi", pcutoff=0.01)

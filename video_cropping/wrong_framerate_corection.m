%% Params
[file,path] = uigetfile('.avi','Select a File', 'E:\videos\paula\');
prompt = {'Ouput video file name:'};
default_input = {''};
answer = inputdlg(prompt, 'Output Name',[1 50], default_input);
output_name = answer{1};

%% Load video and writer object
vid1=VideoReader([path file]);
nframes = vid1.NumberofFrames
writerObj1 = VideoWriter([path output_name]);
writerObj1.FrameRate = 30;
open(writerObj1);

%% Crop each frame
for i=1:30:126001
  im=read(vid1,i);
  imc=im;
  writeVideo(writerObj1,imc);
end
close(writerObj1)


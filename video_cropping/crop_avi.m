%% Params
[file,path] = uigetfile('.avi','Select a File', 'E:\videos\paula\');
prompt = {'Ouput video file name:'};
default_input = {''};
answer = inputdlg(prompt, 'Output Name',[1 50], default_input);
output_name = answer{1};

%% Load video and writer object
vid1=VideoReader([path file]);
n=vid1.NumberOfFrames;
writerObj1 = VideoWriter([path output_name]);
writerObj1.FrameRate = 30;
open(writerObj1);

%% Get cropping dimensions
i = 1;
im=read(vid1,i);
imshow(im);
title('Get Crop Dimensions');
h = imrect(gca);
croprect = wait(h);
close

%% Crop each frame
for i=1:1:n
  im=read(vid1,i);
  imc=imcrop(im, croprect);
  writeVideo(writerObj1,imc);
end
close(writerObj1)


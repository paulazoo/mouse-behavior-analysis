%% Params
[file,path] = uigetfile('.avi','Select a File', '\\anastasia\data\videos\paula\');
prompt = {'Ouput area name:'};
default_input = {''};
answer = inputdlg(prompt, 'Area Name',[1 50], default_input);
output_name = answer{1};

%% Load video and writer object
vid1=VideoReader([path file]);

%% Get area dimensions
i = 1;
im=read(vid1,i);
imshow(im);
title('Get Area');
h = imrect(gca);
croprect = wait(h);
croprect(croprect < 0) = 0;
close

%% Display in command window
format('shortG')
disp(croprect)

%% Save area croprect into a text file
[p,f,e]=fileparts(file);
file_name=fullfile(p,f);
save([path file_name '_' output_name '.mat'], 'croprect')

%% text file
% fileID = fopen([path file_name '_' output_name '.txt'],'w');
% fprintf(fileID,'%.0f, %.0f, %.0f, %.0f', croprect);
% fclose(fileID);

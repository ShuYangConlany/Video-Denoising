% clear all; close all;clc;

groundtruthPath ='C:\Users\Yang Shu\Desktop\fast-Non-local-Means-and-Asymptotic-Non-local-Means-master\FNLM output var=50';
imageFiles = dir(fullfile(groundtruthPath, '*.png'));
fileNames = {imageFiles.name};
sortedIndices = naturalSort(fileNames);
fullPath = fullfile(groundtruthPath, sortedIndices{1});
img = imread(fullPath);  
[height,width,channel]=size(img);
groundtruth=zeros(height,width,length(sortedIndices));
mean=0;
variance=0.2;
variance=0.01;
for i = 1:length(sortedIndices)
    fullPath = fullfile(groundtruthPath, sortedIndices{i});
    img = imread(fullPath);  
    img=double(img)/255;
    groundtruth(:,:,i) = img;
end
images=groundtruth;

folderPath = 'results';

if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

outputVideo = VideoWriter(fullfile(folderPath, 'FNLM output sigma=50'),'MPEG-4');
outputVideo.Quality = 100;
open(outputVideo);
% images=noisy_imgs;
for i = 1:length(images(1,1,:))
%     img = imread(fullfile(folderPath, images(i)));
    writeVideo(outputVideo, images(:,:,i));
end

close(outputVideo);


function sortedFilenames = naturalSort(filenames)
    numbers = regexp(filenames, '\d+', 'match');
    numbers = str2double([numbers{:}]);
    [~, order] = sort(numbers);
    sortedFilenames = filenames(order);
end

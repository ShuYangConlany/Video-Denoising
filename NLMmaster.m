close all; clc; clear all;
% load('missa_noisy_chop.mat')
% load('missa ground truth chop.mat')
load('bird ground truth.mat')
load('30 birds noisy.mat')
input=noisy_imgs(:,:,1);

variance=0.0015;
variance=0.0062;
% variance=0.0138;
% variance=0.0246;
% variance=0.0384;

mean=0;
img=groundtruth(:,:,1);
fullPath="C:\Users\Yang Shu\Desktop\camera.png";
img = double(imread(fullPath))/255;
input=imnoise(img, 'gaussian', mean, variance);
figure(143)
imshow(img)
figure(144)
imshow(input)
%% test results for missa
%FNLM

% output=NLmeansfilter(input,1,7,0.3);
% figure()
% imshow(output)

%
% close all
% input=noisy_img();
tic
output=FNLM(input,2,9,0.08);%10,camera:1-7-0.1;20,camera:2-15-0.07;...
% output=FNLM(input,1,7,0.20);%10,camera:2-15-0.04;20,camera:2-15-0.07;...
% 30,camera:1-15-0.15;40,camera:1-15-0.28;50,camera:1-15-0.20

toc;
elapsedTime = toc;
% img=groundtruth(:,:,5);
psnrFNLM=psnr(output,img,1);
SSIMFNLM=getSSIM(output,img);

fprintf('running time: %.2f sec\n', elapsedTime);
% psnrFNLM=psnr(output,img,1);
% SSIMFNLM=getSSIM(output,img);
figure(111)
imshow(output)


%%
% close all
% input=noisy_img;
tic
[output,W1]=ANLM(input,1,7,0.04);%10,camera:2-7-0.018;20,camera:1-7-0.04;...
% 30,camera:1-7-0.06;40,camera:1-7-0.08;50,camera:1-7-0.08
output=ANLM1(output,1,7,W1);
toc;
elapsedTime = toc;
fprintf('running time: %.2f sec\n', elapsedTime);
psnrA=psnr(output,img,1);
SSIMANLM=getSSIM(output,img);
figure()
imshow(output)

%% test results on birds
% NLM
% load('kalman denoised images.mat')
xxx=input;


denoised_imgs=zeros(size(noisy_imgs));
psnrFNLM=zeros(1,1);
SSIMFNLM=zeros(1,1);
% psnrFNLM=zeros(1,length(groundtruth(1,1,:)));
% SSIMFNLM=zeros(1,length(groundtruth(1,1,:)));

for i=1:1
    output=FNLM(xxx,2,15,0.0015);
    psnrFNLM(i)=psnr(output,img,1);
    SSIMFNLM(i)=getSSIM(output,img);
end


% for i=1:60
% %     output=FNLM(denoisedVideo(:,:,i),1,3,0.1);
%     output=FNLM(noisy_imgs(:,:,i),2,15,0.14);
%     denoised_imgs(:,:,i)=output;
%     psnrFNLM(i)=psnr(output,groundtruth(:,:,i),1);
%     SSIMFNLM(i)=getSSIM(output,groundtruth(:,:,i));
% end

figure()
imshow(output)
% folderName = 'FNLM output var=10';
% if ~exist(folderName, 'dir')
%     mkdir(folderName);
% end
% 
% for i = 1:length(denoised_imgs(1,1,:))
%     fileName = fullfile(folderName, sprintf('image_%d.png', i));
%     imwrite( denoised_imgs(:,:,i),fileName);
% end

%%
% Kalman NLM
% load('kalman denoised images.mat')
denoised_imgs=zeros(size(noisy_imgs));
psnrKal=zeros(1,length(groundtruth(1,1,:)));
SSIMKal=zeros(1,length(groundtruth(1,1,:)));

for i=1:60
%     output=FNLM(denoisedVideo(:,:,i),1,3,0.1);
    output=Kalman_Stack_Filter(noisy_imgs(:,:,i),0.8,0.2);
    output = wiener2(output, [7, 7]);
    output = wiener2(output, [3, 3]);

    denoised_imgs(:,:,i)=output;
    psnrKal(i)=psnr(output,groundtruth(:,:,i),1);
    SSIMKal(i)=getSSIM(output,groundtruth(:,:,i));
end

folderName = 'Kalman output var=50';
if ~exist(folderName, 'dir')
    mkdir(folderName);
end

for i = 1:length(denoised_imgs(1,1,:))
    fileName = fullfile(folderName, sprintf('image_%d.png', i));
    imwrite( denoised_imgs(:,:,i),fileName);
end


%%
denoised_imgs=zeros(size(noisy_imgs));
psnrANLM=zeros(1,length(groundtruth(1,1,:)));
SSIMANLM=zeros(1,length(groundtruth(1,1,:)));

for i=1:60
    [output,W1]=ANLM(noisy_imgs(:,:,i),1,15,0.02);
%      [output,W1]=ANLM(noisy_imgs(:,:,i),2,15,0.05);
%       [output,W1]=ANLM(xxx,2,15,0.15);

%     figure()
%     imshow(output)
    output=ANLM1(output,2,15,W1);
%     figure()
%     imshow(output)
    denoised_imgs(:,:,i)=output;
    psnrANLM(i)=psnr(output,groundtruth(:,:,i),1);
    SSIMANLM(i)=getSSIM(output,groundtruth(:,:,i));

end

folderName = 'ANLM output var=10';

if ~exist(folderName, 'dir')
    mkdir(folderName);
end

for i = 1:length(denoised_imgs(1,1,:))
    fileName = fullfile(folderName, sprintf('image_%d.png', i));
    imwrite( denoised_imgs(:,:,i),fileName);
end

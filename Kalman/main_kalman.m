clear all; close all;clc;

groundtruthPath ='C:\Users\Yang Shu\Desktop\birdP';
imageFiles = dir(fullfile(groundtruthPath, '*.tif'));
fileNames = {imageFiles.name};
sortedIndices = naturalSort(fileNames);
fullPath = fullfile(groundtruthPath, sortedIndices{1});
img = imread(fullPath);  
[height,width,channel]=size(img);
groundtruth=zeros(height,width,length(sortedIndices));
noisy_imgs=zeros(height,width,length(sortedIndices));
mean=0;
% variance=0.01;
variance=0.0361;
% variance=0.0015;
for i = 1:length(sortedIndices)
    fullPath = fullfile(groundtruthPath, sortedIndices{i});
    img = imread(fullPath);  
    img=double(rgb2gray(img));
    img=img/255;
    groundtruth(:,:,i) = img;
    noisy_imgs(:,:,i) = imnoise(img, 'gaussian', mean, variance);
end

figure(122)
imshow(noisy_imgs(:,:,8));


Kalman_psnr=calculate_psnr(groundtruth,noisy_imgs);
title(["PSNR = "+num2str(Kalman_psnr)])

figure(123)
denoisedVideo=Kalman_Stack_Filter(noisy_imgs(:,:,1:5),0.8,0.05);
imshow(denoisedVideo(:,:,31));
Kalman_psnr=psnr(groundtruth,denoisedVideo,1);
title(["PSNR = "+num2str(Kalman_psnr)])

% folderName = 'KalmanOutputMC';
% 
% if ~exist(folderName, 'dir')
%     mkdir(folderName);
% end
% 
% for i = 1:length(denoisedVideo(1,1,:))
%     fileName = fullfile(folderName, sprintf('image_%d.png', i));
%     imwrite( denoisedVideo(:,:,i),fileName);
% end





%% combine NLM
% img=denoisedVideo(:,:,8);
noisy_img=noisy_imgs(:,:,8);
% big var
h=0.25;
% small var
h=0.1;
patch_size=3;
search_window=7;
psnrNLM=zeros(1,length(patch_size(1,1,:)));
denoisedVideoNLM=zeros(size(denoisedVideo));
% img_padded=padarray(img, [patch_size, patch_size], 'symmetric');
% for i =31:length(denoisedVideo(1,1,:))
for i =8:8
    noisy_img_padded=padarray(denoisedVideo(:,:,i), [patch_size, patch_size], 'symmetric');
    temp=NLMAlgorithm(noisy_img_padded,patch_size,search_window,h);
    temp=crop(temp,patch_size);
    psnrNLM(i)=calculate_psnr(groundtruth(:,:,i),temp);
    denoisedVideoNLM(:,:,i)=temp;
end
% 
% folderName = 'KalmanOutputNLM';
% 
% if ~exist(folderName, 'dir')
%     mkdir(folderName);
% end
% 
% for i =31:length(denoisedVideoNLM(1,1,:))
%     fileName = fullfile(folderName, sprintf('image_%d.png', i));
%     imwrite( denoisedVideoNLM(:,:,i),fileName);
% end

%% combine PCA_NLM
h=0.5;
h=0.3;

patch_size=3;
search_window=7;
% [img,img_padded,noisy_img,noisy_img_padded]=dataInit(imgDir(imgIndex),patch_size,GaussianMean,variance,noise_density,noises(noise_type));
psnrPCA=zeros(1,length(patch_size(1,1,:)));
denoisedVideoPCA=zeros(size(denoisedVideo));

for i =4:4
%     length(denoisedVideo(1,1,:))
    noisy_img_padded=padarray(denoisedVideo(:,:,i), [patch_size, patch_size], 'symmetric');
    temp=PCA_NLM(noisy_img_padded, search_window, patch_size, h);
    temp=crop(temp,patch_size);
    psnrNLM(i)=calculate_psnr(groundtruth(:,:,i),temp);
    denoisedVideoPCA(:,:,i)=temp;
end

% folderName = 'KalmanOutputPCA';
% 
% if ~exist(folderName, 'dir')
%     mkdir(folderName);
% end
% 
% for i =1:length(denoisedVideoNLM(1,1,:))
%     fileName = fullfile(folderName, sprintf('image_%d.png', i));
%     imwrite( denoisedVideoPCA(:,:,i),fileName);
% end


% figure(50)
% subplot(3,2,5)
% imshow(cropped_denoised_img)
% psnr_val=calculate_psnr(img,cropped_denoised_img);
% title([algorithm_names(3),"PSNR = "+num2str(psnr_val)])


%% help functions

function sortedFilenames = naturalSort(filenames)
    numbers = regexp(filenames, '\d+', 'match');
    numbers = str2double([numbers{:}]);
    [~, order] = sort(numbers);
    sortedFilenames = filenames(order);
end




function psnr_value = calculate_psnr(I, K)
    % 确保图像是双精度浮点数
    I = double(I);
    K = double(K);

    % 计算均方误差(MSE)
    mse = mean((I(:) - K(:)) .^ 2);

    % 如果MSE为0，则PSNR是无穷大
    if mse == 0
        psnr_value = Inf;
        return;
    end

    % 因为图像的值是在0到1之间，所以MAX_I是1
    MAX_I = 1.0;

    % 计算PSNR
    psnr_value = 10 * log10(MAX_I / sqrt(mse));
end

function denoised_img = NLMAlgorithm(img, patch_size, search_window, h)

    [M, N] = size(img);
    pad_size = floor(patch_size/2);
    offset = floor(search_window/2);

    padded_img = padarray(img, [pad_size, pad_size], 'symmetric');
    denoised_img = zeros(M, N);

    for i = 1+pad_size:M+pad_size
        for j = 1+pad_size:N+pad_size
            P = padded_img(i-pad_size:i+pad_size, j-pad_size:j+pad_size);

            average = 0;
            w_sum = 0;
            
            for k = max(1+pad_size, i-offset):min(M+pad_size, i+offset)
                for l = max(1+pad_size, j-offset):min(N, j+offset)
                    if k == i && l == j
                        continue;
                    end
                    Q = padded_img(int32(k-pad_size):int32(k+pad_size), int32(l-pad_size):int32(l+pad_size));
                    d = sum(sum((P - Q).^2));
                    w = exp(-d/h^2);

                    average = average + w * padded_img(k, l);
                    w_sum = w_sum + w;
                end
            end
            
            denoised_img(i-pad_size, j-pad_size) = average/w_sum;
        end
    end
end

function [cropped_denoised_img] = crop(denoised_img,patch_size)
    [h,w]=size(denoised_img);
    cropped_denoised_img=denoised_img(patch_size+1:h-patch_size,1+patch_size:w-patch_size);
end


function denoised_img = PCA_NLM(img, searchWindowSize, patchSize, h)
    % img: 输入的噪声图像
    % searchWindowSize: 搜索窗口的大小 (例如: 21x21)
    % patchSize: 图像块的大小 (例如: 7x7)
    % h: 过滤器参数
    % m: 用于PCA的主成分数量
    [rows, cols] = size(img);
    denoised_img = zeros(rows, cols);
    halfSearch = floor(searchWindowSize/2);
    halfPatch = floor(patchSize/2);
    for i = 1+patchSize:rows-patchSize
        for j = 1+patchSize:cols-patchSize
            
            searchWindow = img(max(i-halfSearch,1):min(i+halfSearch,rows), max(j-halfSearch,1):min(j+halfSearch,cols));
            referencePatch = img(max(i-halfPatch,1):min(i+halfPatch,rows), max(j-halfPatch,1):min(j+halfPatch,cols));
            
            
            dataMatrix = zeros(patchSize^2,(size(searchWindow, 1) - patchSize + 1)^2);  
            for x = 1:size(searchWindow, 1) - patchSize + 1
                for y = 1:size(searchWindow, 2) - patchSize + 1
                    patch = searchWindow(x:x+patchSize-1, y:y+patchSize-1);
                    dataMatrix(:,(x-1)*(size(searchWindow, 1) - patchSize + 1)+y) =  patch(:);  
                end
            end
            [coeff, ~, latent] = pca(dataMatrix');  
            % decide m
            total_variance = sum(latent);
            cumulative_variance = 0;
            m = 0;
            for ptr = 1:length(latent)
                cumulative_variance = cumulative_variance + latent(ptr);
                if cumulative_variance >= total_variance * 0.90
                    m = ptr;
                    break;
                end
            end

            transformMatrix = coeff(:, 1:m);
            referenceProjection = referencePatch(:)' * transformMatrix;
            
            weights = zeros(size(dataMatrix, 2), 1);
            patchValues=0;
            for k = 1:size(dataMatrix, 2)
                projection = dataMatrix(:, k)' * transformMatrix;
                distance = norm(referenceProjection - projection)^2;
                weights(k) = exp(-distance/h^2);
                patchValues=patchValues+searchWindow(halfPatch+1+idivide(int32(k),int32(1+(size(searchWindow, 1) - patchSize + 1))),...
                    halfPatch+1+mod(k-1,(size(searchWindow, 1) - patchSize + 1)))*weights(k);
            end
            
            denoised_img(i,j) = patchValues/ sum(weights);  

%             weights = weights / sum(weights);
%             patchValues = sum(dataMatrix * weights, 2);
%             denoised_img(i,j) = mean(patchValues);  
        end
    end
end
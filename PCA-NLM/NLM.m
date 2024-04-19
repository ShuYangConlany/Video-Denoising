% NLP Algorithm
clc;clear all;
close all;
% for Gaussian Noise
GaussianMean=0;
variance=0.11;
% for Salt&pepper Noise
noise_density=0.05;
% for NLM Algorithm
% h=0.3;
% for PCA_NLM Algorithm
m=7;
% for texture detecting function
B=1;
spikeAvoidPercentage=0.2;


imgDir=["missa_80.tif","Fall_trees_0.5.tif","coastguard_90.tif"];
noises=["Gaussian";"salt&pepper";"striped";"color noise"];
algorithm_names=["Wiener","NLM","PCA-NLM","textured PCA-NLM"];
noise_type=1;
imgIndex=1;

%% Wiener Algorithm
patch_size=3;
search_window=7;
[img,img_padded,noisy_img,noisy_img_padded]=dataInit(imgDir(imgIndex),patch_size,GaussianMean,variance,noise_density,noises(noise_type));
denoised_img = wiener2(noisy_img, [3, 3]);
% resultPlot(img,noisy_img,img_padded,denoised_img,algorithm_names(1))
figure(50)
subplot(3,2,1)
imshow(img)
title("original image")
subplot(3,2,2)
imshow(noisy_img)
psnr_val=calculate_psnr(img,noisy_img);
title(["noisy image","PSNR = "+num2str(psnr_val)])
subplot(3,2,3)
imshow(denoised_img)
psnr_val=calculate_psnr(img,denoised_img);
title([algorithm_names(1),"PSNR = "+num2str(psnr_val)])

%% original NLM Algorithm

h=0.3;
h=0.3;
patch_size=5;
search_window=45;
[img,img_padded,noisy_img,noisy_img_padded]=dataInit(imgDir(imgIndex),patch_size,GaussianMean,variance,noise_density,noises(noise_type));
tic
denoised_img=NLMAlgorithm(noisy_img_padded,patch_size,search_window,h);
toc;
elapsedTime = toc;
fprintf('running time: %.2f sec\n', elapsedTime);

chopped_denoised_img=chop(denoised_img,patch_size);
% resultPlot(img,noisy_img,img_padded,cropped_denoised_img,algorithm_name)


figure(50)
subplot(3,2,4)
imshow(chopped_denoised_img)
psnr_val=calculate_psnr(chopped_denoised_img,img);
title([algorithm_names(2),"PSNR = "+num2str(psnr_val)])

%% PCA_NLM Algorithm
h=0.5;
patch_size=3;
search_window=9;
[img,img_padded,noisy_img,noisy_img_padded]=dataInit(imgDir(imgIndex),patch_size,GaussianMean,variance,noise_density,noises(noise_type));
denoised_img=PCA_NLM(noisy_img_padded, search_window, patch_size, h);
chopped_denoised_img=chop(denoised_img,patch_size);
% resultPlot(img,noisy_img,img_padded,cropped_denoised_img,algorithm_name)

figure(50)
subplot(3,2,5)
imshow(chopped_denoised_img)
psnr_val=psnr(chopped_denoised_img,img,1);
title([algorithm_names(3),"PSNR = "+num2str(psnr_val)])

%% PCA_NLM Algorithm adding the texture analysis
h=0.3;
patch_size=5;
search_window=13;
patchSizeLarge=5;
patchSizeSmall=3;
hLarge=0.3;
hSmall=0.3;
[img,img_padded,noisy_img,noisy_img_padded]=dataInit(imgDir(imgIndex),patch_size,GaussianMean,variance,noise_density,noises(noise_type));
textureFunction = textureDetectingFunction(noisy_img,B);

normalizedTextureFunction = zeros(size(textureFunction));
normalizedTextureFunction( textureFunction < 3*10^(-5)) = 1;
normalizedTextureFunction(textureFunction >= 3*10^(-5) & textureFunction < 10^(-4)) = 100;
normalizedTextureFunction(textureFunction >= 10^(-4)) = 250;
figure()
imshow(normalizedTextureFunction/255)
% showTextureFunction(textureFunction)
denoised_img=textured_PCA_NLM(noisy_img_padded, search_window, patch_size, h, normalizedTextureFunction,spikeAvoidPercentage,patchSizeLarge,patchSizeSmall,hLarge,hSmall);
chopped_denoised_img=chop(denoised_img,patch_size);
resultPlot(img,noisy_img,img_padded,chopped_denoised_img,algorithm_name)

figure(50)
subplot(3,2,6)
imshow(chopped_denoised_img)
psnr_val=psnr(chopped_denoised_img,img,1);
title([algorithm_names(4),"PSNR = "+num2str(psnr_val)])

function [] = showTextureFunction(textureFunction)
    A_min = min(textureFunction(:));
    A_max = max(textureFunction(:)); 
    
    A_normalized = (textureFunction - A_min) / (A_max - A_min);
    
    A_scaled = uint8(255 * A_normalized);
    B = zeros(size(A_scaled));
    
    B(A_scaled >= 0 & A_scaled <= 1) = 0;    
    B(A_scaled > 1 & A_scaled <= 100) = 1; 
    % B(A_scaled > 100 & A_scaled <= 255) = 3; 
    
    cmap = [
        0 0 0;
        0 0 0.8;
%         1 0 0;
    ];
    

    figure()
    imagesc(B);
    colormap(cmap);
    axis('image');
    colorbar;  
end

function [cropped_denoised_img] = chop(denoised_img,patch_size)
    [h,w]=size(denoised_img);
    cropped_denoised_img=denoised_img(patch_size+1:h-patch_size,1+patch_size:w-patch_size);
end

function [img,img_padded,noisy_img,noisy_img_padded] = dataInit(imgDir,patch_size,mean,variance,noise_density,noise_type)
    img=im2double(imread(imgDir));
    img_padded=padarray(img, [patch_size, patch_size], 'symmetric');
    switch noise_type
        case 'Gaussian'
            noisy_img = imnoise(img, 'gaussian', mean, variance);
        case 'salt&pepper'
            noisy_img = imnoise(img, 'salt & pepper', noise_density);

    end
    noisy_img_padded=padarray(noisy_img, [patch_size, patch_size], 'symmetric');
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

function []=resultPlot(img,noisy_img,img_padded,denoised_img,algorithm_name)
    figure()
    subplot(2,2,1)
    imshow(img)
    title("original img")
    subplot(2,2,2)
    imshow(noisy_img)
    psnr_val=calculate_psnr(img,noisy_img);
    title(["noisy img psnr=",num2str(psnr_val)])
    subplot(2,2,3)
    imshow(denoised_img)
    psnr_val=psnr(img,denoised_img);
    title([algorithm_name,"denoising psnr=",num2str(psnr_val)])
    e=(img-denoised_img);
    subplot(2,2,4)
    imshow(e)
    title([algorithm_name,"error img"])
end

function psnr_val = psnr(original_img, reconstructed_img)
    if size(original_img) ~= size(reconstructed_img)
        error('The input images must be of the same size');
    end

    % Convert images to double precision
    original = double(original_img);
    reconstructed = double(reconstructed_img);
    
    % Calculate MSE
    mse = mean((original - reconstructed).^2, 'all');

    max_pixel = 1;

    % Calculate PSNR
    psnr_val = 20 * log10(max_pixel / sqrt(mse));
end

function denoised_img = PCA_NLM(img, searchWindowSize, patchSize, h)
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
                %%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% TODO
                %%%%%%%%%%%
                distance = norm(referenceProjection - projection);
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

function textureFunction = textureDetectingFunction(img,B)
    [gx, gy] = gradient(img);
    [rows, cols]=size(img);
    Ixx = gx.^2;
    Iyy = gy.^2;
    Ixy = gx.*gy;
    
    windowSize = 3;
    w = fspecial('average', windowSize);
    
    Ixx = conv2(Ixx, w, 'same');
    Iyy = conv2(Iyy, w, 'same');
    Ixy = conv2(Ixy, w, 'same');
    
    eig1 = zeros(rows, cols);
    eig2 = zeros(rows, cols);
    textureFunction=zeros(rows, cols);
    for r = 1:rows
        for c = 1:cols
            A = [Ixx(r,c), Ixy(r,c); Ixy(r,c), Iyy(r,c)];
            eigenvalues = eig(A);
            max_eig=max(eigenvalues(1),eigenvalues(2));
            min_eig=min(eigenvalues(1),eigenvalues(2));
            eig1(r,c) = max_eig;
            eig2(r,c) = min_eig;
            textureFunction(r,c)=abs(max_eig-min_eig)*max_eig/(1+B*(max_eig+min_eig));
        end
    end
end

function denoised_img = textured_PCA_NLM(img, searchWindowSize, patchSize, h, textureFunction,spikeAvoidPercentage,patchSizeLarge,patchSizeSmall,hLarge,hSmall)
    [rows, cols] = size(img);
    denoised_img = zeros(rows, cols);
    halfSearch = floor(searchWindowSize/2);
    halfPatch = floor(patchSize/2);
    sortedTextureFunction = sort(textureFunction(:), 'descend');
    index = ceil(spikeAvoidPercentage * length(sortedTextureFunction));
    value = sortedTextureFunction(index);
    C=1/value;
    for i = 1+patchSizeLarge:rows-patchSizeLarge
        for j = 1+patchSizeLarge:cols-patchSizeLarge
            if textureFunction(i-patchSizeLarge,j-patchSizeLarge)==250
                patchSize=patchSizeSmall;
                halfPatch=floor(patchSize/2);
                h=hSmall;
            elseif textureFunction(i-patchSizeLarge,j-patchSizeLarge)==100
                patchSize=patchSizeLarge;
                halfPatch=floor(patchSize/2);
                h=hSmall;
            else
                patchSize=patchSizeLarge;
                halfPatch=floor(patchSize/2);
                h=hLarge;
            end
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
%             if abs(C*textureFunction(i,j))>0.99
%                 m=ceil(patchSize^2)-1;
%             elseif  abs(C*textureFunction(i,j))<0.1
%                 m=ceil(patchSize^2*0.1);
%             else
%                 m=ceil(abs(C*textureFunction(i,j))*patchSize^2);
%             end

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

        end
    end
end


function psnr_value = calculate_psnr(I, K)
    I = double(I);
    K = double(K);

    mse = mean((I(:) - K(:)) .^ 2);

    if mse == 0
        psnr_value = Inf;
        return;
    end

    MAX_I = 1;

    psnr_value = 10 * log10(MAX_I / mse);
end

% figure()
% imshow(I/255)
% figure()
% imshow(K/255)

% figure()
% imshow(img)
% figure()
% imshow(chopped_denoised_img)

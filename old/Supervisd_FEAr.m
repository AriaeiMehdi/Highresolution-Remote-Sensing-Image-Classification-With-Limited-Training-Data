clc
clear all


%% Model
lgraph = layerGraph();

tempLayers = [
    imageInputLayer([17 17 3],"Name","imageinput")
    convolution2dLayer([3 3],64,"Name","conv_A1","Padding","same")
    reluLayer("Name","relu_A1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_A2","Padding","same")
    reluLayer("Name","relu_A2")
    convolution2dLayer([3 3],64,"Name","conv_A3","Padding","same")
    reluLayer("Name","relu_A3")
    convolution2dLayer([3 3],64,"Name","conv_A4","Padding","same")
    reluLayer("Name","relu_A4")
    convolution2dLayer([3 3],64,"Name","conv_A5","Padding","same")
    reluLayer("Name","relu_A5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat1")
    globalAveragePooling2dLayer("Name","gapool")
    flattenLayer("Name","flatten")
    fullyConnectedLayer(128,"Name","Spa_fc")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    featureInputLayer(200,"Name","featureinput")
    fullyConnectedLayer(128,"Name","Spe_fc")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(1,2,"Name","fc1")
    reluLayer("Name","relu_fc1")
    fullyConnectedLayer(128,"Name","fc2")
    reluLayer("Name","relu_fc2")
    fullyConnectedLayer(128,"Name","fc3")
    reluLayer("Name","relu_fc3")
    fullyConnectedLayer(17,"Name","fc4")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"relu_A1","conv_A2");
lgraph = connectLayers(lgraph,"relu_A1","concat1/in2");
lgraph = connectLayers(lgraph,"relu_A5","concat1/in1");
lgraph = connectLayers(lgraph,"Spe_fc","fc1/in2");
lgraph = connectLayers(lgraph,"Spa_fc","fc1/in1");

plot(lgraph);
  analyzeNetwork(lgraph);  % figure; plot(lgraph)

%% Load data

% Input image 
load Indian_pines_corrected.mat;    % load data
load Indian_pines_gt.mat;           % load GTM
 
[h w] = size(indian_pines_gt);
gt = uint8(indian_pines_gt);
%% Min&Max Scale the image
% scale the image 
scaled_img = [];
[~,~, num_bands] = size(indian_pines_corrected);
for i = 1:num_bands
    min_val = min(indian_pines_corrected(:, :, i), [], 'all');
    max_val = max(indian_pines_corrected(:, :, i), [], 'all');

    % Scale the image to the range [0, 255]
    temp = 255 * (double(indian_pines_corrected(:, :, i)) - min_val) / (max_val - min_val);

    % Convert the scaled image to uint8 format
    scaled_img = cat(3, scaled_img, uint8(temp));
end 

indian_pines_corrected = double(scaled_img)/255;

clearvars -except indian_pines_corrected lgraph indian_pines_gt h w gt
%% PCA Transform

PC_num = 3;
IMG = pca_Trans(indian_pines_corrected, PC_num);

clearvars PC_num
%% Extract Patch 

patch_size = 17;
IMG = padarray(IMG,[(patch_size-1)/2 (patch_size-1)/2],0,'both');
indian_pines_corrected = padarray(indian_pines_corrected,[(patch_size-1)/2 (patch_size-1)/2],0,'both');
indian_pines_gt = padarray(indian_pines_gt,[(patch_size-1)/2 (patch_size-1)/2],0,'both');


[height, width, depth] = size(IMG);

patches = zeros(patch_size, patch_size, depth, (height-patch_size+1)*(width-patch_size+1));
pixels = zeros(200, 1, (height-patch_size+1)*(width-patch_size+1),1);
GTM = categorical(zeros((height-patch_size+1)*(width-patch_size+1),1));

k = 1;
for i = 1:height - patch_size+1
    for j = 1:width - patch_size+1
        patch = IMG(i:(i+patch_size-1), j:(j+patch_size-1), :);
        patches(:, :, :, k) = patch;
        pixel = indian_pines_corrected(i-1+(1+patch_size)/2, j-1+(1+patch_size)/2, :);
        pixels(:, :, k) = pixel;
        gtm = indian_pines_gt(i-1+(1+patch_size)/2, j-1+(1+patch_size)/2, :);
        GTM(j+(i-1)*(height-patch_size+1)) = categorical(gtm);
        k = k +1;
    end
end

clearvars depth gtm height i IMG indian_pines_gt indian_pines_corrected j k patch patch_size pixel width
%% Train And Test Data

dspatches = arrayDatastore(patches, IterationDimension=4);
dspixels = arrayDatastore(pixels(:,1,:), IterationDimension=3);
dsGTM = arrayDatastore(GTM);

Dataset = combine(dspatches,dspixels,dsGTM);

Train_percent = 0.6;
idxs = splitlabels(Dataset,Train_percent,'randomized',UnderlyingDatastoreIndex=3);

dsTrain = subset(Dataset,idxs{1});
dsVal = subset(Dataset,idxs{2});


% Dataset and trainset information 
Dataset_info = countlabels(Dataset,'UnderlyingDatastoreIndex',3);
dsTrain_info = countlabels(dsTrain,'UnderlyingDatastoreIndex',3);

fig1 = uifigure(Name="Dataset Info");
uitable(fig1,"Data",Dataset_info, "Position",[20 20 500 350]);
fig2 = uifigure(Name="TrainSet Info");
uitable(fig2,"Data",dsTrain_info, "Position",[20 20 500 350]);


clearvars -except dsTrain lgraph dsVal Dataset h w gt
%% Train The Network

options = trainingOptions('adam', ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.2, ...
    LearnRateDropPeriod=5, ...
    Shuffle="every-epoch",...
    MaxEpochs=10,...
    MiniBatchSize = 32,...
    ValidationFrequency=1000, ...
    ValidationData = dsVal, ...
    Verbose=0,Plots= 'training-progress');


trained = trainNetwork(dsTrain,lgraph,options);

% lgraph = layerGraph(layers);
analyzeNetwork(lgraph);
%% Network Evaluation

Result = uint8(classify(trained,Dataset));
Result = reshape(Result,[h, w])';
imagesc(Result);
figure;
imagesc(gt);

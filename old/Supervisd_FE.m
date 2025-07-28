clc
clear all

%% Load data

% Input image 
load Indian_pines_corrected.mat;    % load data
load Indian_pines_gt.mat;           % load GTM

GTM = uint8(indian_pines_gt);
HS_ORG = indian_pines_corrected;
[Nr, Nc, Nb] = size(HS_ORG);
Nclass = unique(GTM);
Nclass = numel(Nclass) - 1;

clear indian_pines_gt indian_pines_corrected
%% Min&Max Scale the image

% scale the image 
HS_scaled = MinMax(HS_ORG); % For uint8 or uint16 use this: MinMax(HS_ORG, "uint8")

%% PCA Transform

PCAnum = 3;
HS_PCA = pca_Trans(HS_scaled, PCAnum);

%% Model
lgraph = layerGraph();
tempLayers = [
    imageInputLayer([17 17 PCAnum],"Name","imageinput")
    convolution2dLayer([3 3],64,"Name","conv_A1","Padding","same")
    reluLayer("Name","relu_A1")
    batchNormalizationLayer("Name","batchnorm")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_A2","Padding","same")
    reluLayer("Name","relu_A2")
    batchNormalizationLayer("Name","batchnorm_1")
    convolution2dLayer([3 3],64,"Name","conv_A3","Padding","same")
    reluLayer("Name","relu_A3")
    batchNormalizationLayer("Name","batchnorm_2")
    convolution2dLayer([3 3],64,"Name","conv_A4","Padding","same")
    reluLayer("Name","relu_A4")
    batchNormalizationLayer("Name","batchnorm_3")
    convolution2dLayer([3 3],64,"Name","conv_A5","Padding","same")
    reluLayer("Name","relu_A5")
    batchNormalizationLayer("Name","batchnorm_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat1")
    globalAveragePooling2dLayer("Name","gapool")
    flattenLayer("Name","flatten")
    fullyConnectedLayer(128,"Name","Spa_fc")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    imageInputLayer([1 1 Nb],"Name","imageinput_1")
    flattenLayer("Name","flatten_1")
    fullyConnectedLayer(128,"Name","Spe_fc")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(1,2,"Name","fc1")
    reluLayer("Name","relu_fc1")
    batchNormalizationLayer("Name","batchnorm_5")
    fullyConnectedLayer(128,"Name","fc2")
    reluLayer("Name","relu_fc2")
    batchNormalizationLayer("Name","batchnorm_6")
    fullyConnectedLayer(128,"Name","fc3")
    reluLayer("Name","relu_fc3")
    batchNormalizationLayer("Name","batchnorm_7")
    fullyConnectedLayer(Nclass,"Name","fc4")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"batchnorm","conv_A2");
lgraph = connectLayers(lgraph,"batchnorm","concat1/in1");
lgraph = connectLayers(lgraph,"batchnorm_4","concat1/in2");
lgraph = connectLayers(lgraph,"Spa_fc","fc1/in1");
lgraph = connectLayers(lgraph,"Spe_fc","fc1/in2");
analyzeNetwork(lgraph)

%% Extract Patch 

Patch_Size = 17;
[patches, ~] = Patch_maker(HS_PCA, Patch_Size, "patch");            % extract patches
[~, center_pixel] = Patch_maker(HS_scaled, Patch_Size, "pixel");    % extract centere pixel from original HS
[~, gtm] = Patch_maker(GTM, Patch_Size, "pixel");    % extract centere pixel from original HS
arryGTM = reshape(gtm, [], 1);

clearvars Patch_Size
%% Train And Test Data


[non_zero_indx,~,~] = find(uint8(arryGTM));

dspatches = arrayDatastore(patches, IterationDimension=4);
dspixels = arrayDatastore(center_pixel, IterationDimension=4);
dsGTM = arrayDatastore(categorical(arryGTM), IterationDimension=1);
Val_Dataset = combine(dspatches,dspixels,dsGTM);
clearvars dspatches dspixels dsGTM

dspatches = arrayDatastore(patches(:,:,:,non_zero_indx), IterationDimension=4);
dspixels = arrayDatastore(center_pixel(:,:,:,non_zero_indx), IterationDimension=4);
dsGTM = arrayDatastore(categorical(arryGTM(non_zero_indx)));
Dataset = combine(dspatches,dspixels,dsGTM);


% Split Dataset to train and validation set
Train_percent = 0.5;
idxs = splitlabels(Dataset,Train_percent,UnderlyingDatastoreIndex=3);

dsTrain = subset(Dataset,idxs{1});
dsVal = subset(Dataset,idxs{2});



% Dataset and trainset information 
dsValidation_info = countlabels(dsVal,'UnderlyingDatastoreIndex',3);
dsTrain_info = countlabels(dsTrain,'UnderlyingDatastoreIndex',3);

fig1 = uifigure(Name="Validation Info");
uitable(fig1,"Data",dsValidation_info, "Position",[20 20 500 350]);
fig2 = uifigure(Name="TrainSet Info");
uitable(fig2,"Data",dsTrain_info, "Position",[20 20 500 350]);


%clearvars -except dsTrain lgraph dsVal Dataset h w gt
%% Train The Network

options = trainingOptions('adam', ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.2, ...
    LearnRateDropPeriod=5, ...
    Shuffle="every-epoch",...
    MaxEpochs=5,...
    MiniBatchSize = 512,...
    ValidationFrequency=100, ...
    ValidationData = dsVal, ...
    Verbose=0,Plots= 'training-progress');


trained = trainNetwork(dsTrain,lgraph,options);
save('trained.mat','trained');

%% Network Evaluation

Result = uint8(classify(trained,Val_Dataset));
Result = Result';
Res = reshape(Result,Nc,Nr).';
imagesc(Res);
figure;
imagesc(uint8(GTM));








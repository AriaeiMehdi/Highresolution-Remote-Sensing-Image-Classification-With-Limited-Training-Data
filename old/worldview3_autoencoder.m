clear all;
clc;

%% WorldView3 Data Set
load Datasets/WV3_36.mat
PAN = double(PAN_ORG);
PAN = MinMax(PAN);
figure;imshow( PAN);
clear PAN_ORG MS_ORG;
%% data patching
patchSize=16; % patch size
imSz = size(PAN);
patchSz = [16 16];
xIdxs = [1:patchSz(2):imSz(2) imSz(2)+1];
yIdxs = [1:patchSz(1):imSz(1) imSz(1)+1];
patches = cell(length(yIdxs)-1,length(xIdxs)-1);
for i = 1:length(yIdxs)-1
    Isub = PAN(yIdxs(i):yIdxs(i+1)-1,:);
    for j = 1:length(xIdxs)-1
        patches{i,j} = Isub(:,xIdxs(j):xIdxs(j+1)-1);
    end
end
PANpatches = reshape( cat(3,patches{:}), [16,16,1,size(patches,1)^2]);
[~,~,~,n] = size(PANpatches) ;
P = 0.80 ;
idx = randperm(n)  ;
PANpatches_Training = PANpatches(:, :, :, idx(1:round(P*n))); 
PANpatches_Testing = PANpatches(:, :, :, idx(round(P*n)+1:end));
%% Datastores 
dsTrain = arrayDatastore(PANpatches_Training, IterationDimension=4);
dsTest = arrayDatastore(PANpatches_Testing, IterationDimension=4);
%% Encoder Network
numLatentChannels = 1024;
imageSize = [16 16 1];
layersE = [
    imageInputLayer(imageSize,Normalization="none")
    convolution2dLayer(3,32,Padding="same",Stride=2)
    reluLayer
    convolution2dLayer(3,32,Padding="same",Stride=2)
    reluLayer
    convolution2dLayer(3,64,Padding="same",Stride=2)
    reluLayer
    convolution2dLayer(3,64,Padding="same",Stride=2)
    reluLayer
    fullyConnectedLayer(2*numLatentChannels)
    samplingLayer];
%% Decoder Network
projectionSize = [1 1 512];
numInputChannels = imageSize(3);
layersD = [
    featureInputLayer(numLatentChannels)
    projectAndReshapeLayer(projectionSize)
    transposedConv2dLayer(3,64,Cropping="same",Stride=2)
    reluLayer
    transposedConv2dLayer(3,64,Cropping="same",Stride=2)
    reluLayer
    transposedConv2dLayer(3,32,Cropping="same",Stride=2)
    reluLayer
    transposedConv2dLayer(3,32,Cropping="same",Stride=2)
    reluLayer
    transposedConv2dLayer(3,numInputChannels,Cropping="same")
    sigmoidLayer];
%%
netE = dlnetwork(layersE);
netD = dlnetwork(layersD);
%% Train Hyperparameters
numEpochs = 150;
miniBatchSize = 512;
learnRate = 1e-4;
%%  prepare mini-batch for train
numOutputs = 1;
mbq = minibatchqueue(dsTrain,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB", ...
    PartialMiniBatch="discard");
%% Initialize train parameters
trailingAvgE = [];
trailingAvgSqE = [];
trailingAvgD = [];
trailingAvgSqD = [];
%%
numObservationsTrain = size(PANpatches_Training,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
%%
monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info="Epoch", ...
    XLabel="Iteration");
%% custom training loop
epoch = 0;
iteration = 0;
% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    % Shuffle data.
    shuffle(mbq);
    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        % Read mini-batch of data.
        X = next(mbq);
        % Evaluate loss and gradients.
        [loss,gradientsE,gradientsD] = dlfeval(@modelLoss,netE,netD,X);
        % Update learnable parameters.
        [netE,trailingAvgE,trailingAvgSqE] = adamupdate(netE, ...
            gradientsE,trailingAvgE,trailingAvgSqE,iteration,learnRate);
        [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, ...
            gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);
        % Update the training progress monitor. 
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100*iteration/numIterations;
    end
end
clearvars -except netD netE
%% WorldView3 Data Set
load Datasets/WV3_36.mat
MS_ORG=double(MS_ORG);
MS_ORG=imresize(MS_ORG,4,"nearest");
GTM =imread('Datasets/WV3_36GTM.bmp');
labelClass=["Concrete","Boat","Car","Tree","Water","Asphalt","Sea","Building","Shelter","Unknown","Soil"];
RGB =imread('Datasets/WV3_36HVS.png');
PAN =imread('Datasets/WV3_36PAN.png');
figure;
imshow( RGB);
[Nr, Nc, nb] = size(PAN_ORG);
[Nr, Nc, Nb] = size(MS_ORG);
Spectral=uint16(reshape(MS_ORG,[Nr*Nc Nb]));
clear PAN1 ;
%% data patching
patchSize=16; % patch size
maxEpochs=5;
PANpatches=dataPatcher2(PAN_ORG,patchSize);
%% create train and test & validation datastores
Train_percent=0.999;%15; % percent of total data select randomly for train data
Label=reshape(GTM,[Nr*Nc,1]); % spectral labels
    [trainInds,trainMap]= Train_Sampels(Spectral,Label,Train_percent);
        num_class = max(trainMap,[],"all");
        numTrain = nnz(trainMap);
        trainMap=reshape(trainMap,Nr,Nc);
            [trainInd,valInd,~] = dividerand(numTrain,0.90,0.10,0);
labelTrain  = zeros(numTrain,1);
trainDataPAN= zeros(patchSize,patchSize,nb,numTrain);
trainDataMS = zeros(1,1,Nb,numTrain);
    i = 0;
    for ix = 1:Nr
        for iy = 1:Nc
            if trainMap(ix,iy) ~= 0
                i = i+1;
                labelTrain(i,1) = trainMap(ix,iy);                
                trainDataPAN(:,:,:,i) = PANpatches(:,:,:,ix,iy);
                trainDataMS(1,1,:,i) = MS_ORG(ix,iy,:);
            end
        end
    end
    PANpatches = reshape(PANpatches,patchSize,patchSize,nb,[]);
spec=zeros(1,1,Nb,Nr*Nc);
spec(1,1,:,:)=Spectral.';
%% Datastores 
    SpatialFeatures = arrayDatastore(PANpatches,IterationDimension=4);
    SpectralFeatures= arrayDatastore(spec,IterationDimension=4);
    SpatialSpectralF = combine(SpatialFeatures,SpectralFeatures);
    label = categorical(labelTrain);
% test datastores 
        dsTesSpa =      arrayDatastore(trainDataPAN(:,:,:,trainInd),IterationDimension=4);
        dsTesSpec =     arrayDatastore(trainDataMS(1,1,:,trainInd),IterationDimension=4);
        dsTeslabel =    arrayDatastore(label(trainInd));
        dsTrainSpa =    combine(dsTesSpa,dsTeslabel);
        dsTrainSpc =    combine(dsTesSpec,dsTeslabel);
        dsTrainSpaSpc = combine(dsTesSpa,dsTesSpec,dsTeslabel);
% validation datastores
    dsVSpa = arrayDatastore(trainDataPAN(:,:,:,valInd),IterationDimension=4);
    dsVspec = arrayDatastore(trainDataMS(1,1,:,valInd),IterationDimension=4);
    dsVlabel = arrayDatastore(label(valInd));
    dsValSpa = combine(dsVSpa,dsVlabel);
    dsValSpc = combine(dsVspec,dsVlabel);
    dsValSS = combine(dsVSpa,dsVspec,dsVlabel);
%%
% Remove the last layer of encoder and add new layers
New_model = removeLayers(netE, 'layer');
tempLayer = [flattenLayer("Name","flatten_autoenc")];
New_model = addLayers(New_model,tempLayer);
New_model = connectLayers(New_model,"fc","flatten_autoenc");
tempLayers = [
    imageInputLayer([1 1 Nb],"Name","Spectral_Nb")
    flattenLayer("Name","flatten")
    fullyConnectedLayer(128,"Name","fullyConct_8_128")
    reluLayer("Name","relu_Spec_8_128");
    fullyConnectedLayer(128,"Name","fullyConct_128_128")
    reluLayer("Name","relu_Spec_128_128");
    fullyConnectedLayer(128,"Name","fullyConct_128")
    reluLayer("Name","relu_Spec_128")];
New_model = addLayers(New_model,tempLayers);
tempLayers = [
        concatenationLayer(1,2,"Name","concat_Spa&Sec")
        % MLP and classification
        fullyConnectedLayer(128,"Name","fullyConct_256_128")
        reluLayer("Name","relu_128")
        fullyConnectedLayer(64,"Name","fullyConct_128_64")
        reluLayer("Name","relu_64")
        fullyConnectedLayer(num_class,"Name","fullyConct_64_Nclass")
        softmaxLayer("Name","softmax")
        classificationLayer("Name","classoutput")];
New_model = addLayers(New_model,tempLayers);
New_model = connectLayers(New_model,"flatten_autoenc","concat_Spa&Sec/in1");
New_model = connectLayers(New_model,"relu_Spec_128","concat_Spa&Sec/in2");
lgraph = layerGraph(New_model);
analyzeNetwork(lgraph);
%%
options = trainingOptions('adam', ...
    LearnRateSchedule = "piecewise", ...
    Shuffle ="every-epoch",...
    MaxEpochs = maxEpochs,...
    MiniBatchSize = 512,...
    ValidationFrequency = 50, ...
    ValidationData = dsValSS, ...
    Verbose=0, Plots = 'training-progress');
clearvars -except netD netE dsTrainSpaSpc lgraph options SpatialSpectralF Label GTM PAN RGB% free up ram
ctnet3 = trainNetwork(dsTrainSpaSpc,lgraph,options);
Res = uint8(classify(ctnet3,SpatialSpectralF));
Classification_Performance(uint8(Label),Res,GTM,PAN,RGB,"DCNN Spatial-Spectral Classification",labelClass);
%% These Functions are needed for Autoencoder part, Do NOT edit or delete
function [loss,gradientsE,gradientsD] = modelLoss(netE,netD,X)   
    % Forward through encoder.
    [Z,mu,logSigmaSq] = forward(netE,X);
    % Forward through decoder.
    Y = forward(netD,Z);
    % Calculate loss and gradients.
    loss = elboLoss(Y,X,mu,logSigmaSq);
    [gradientsE,gradientsD] = dlgradient(loss,netE.Learnables,netD.Learnables);
end
%%
function loss = elboLoss(Y,T,mu,logSigmaSq)
    % Reconstruction loss.
    reconstructionLoss = mse(Y,T);
    % KL divergence.
    KL = -0.5 * sum(1 + logSigmaSq - mu.^2 - exp(logSigmaSq),1);
    KL = mean(KL);
    % Combined loss.
    loss = reconstructionLoss + KL;
end
%%
function Y = modelPredictions(netE,netD,mbq)
    Y = [];
    % Loop over mini-batches.
    while hasdata(mbq)
        X = next(mbq);
        % Forward through encoder.
        Z = predict(netE,X);
        % Forward through dencoder.
        XGenerated = predict(netD,Z);
        % Extract and concatenate predictions.
        Y = cat(4,Y,extractdata(XGenerated));
    end
end
%%
function X = preprocessMiniBatch(dataX)
    % Concatenate.
    X = cat(4,dataX{:});
end
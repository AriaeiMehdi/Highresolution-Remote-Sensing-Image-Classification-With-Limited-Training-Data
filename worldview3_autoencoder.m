clear all;
clc;

%% WorldView3 Data Set
load Datasets1/WV3_36.mat
MS_ORG = double(MS_ORG);
MS_ORG = MinMax(MS_ORG);
MS_ORG1=imresize(MS_ORG,4,"nearest");

load Datasets1/WV3_15.mat
MS_ORG = double(MS_ORG);
MS_ORG = MinMax(MS_ORG);
MS_ORG2=imresize(MS_ORG,4,"nearest");

load Datasets1/WV3_14.mat
MS_ORG = double(MS_ORG);
MS_ORG = MinMax(MS_ORG);
MS_ORG3=imresize(MS_ORG,4,"nearest");

load Datasets1/WV3_00.mat
MS_ORG = double(MS_ORG);
MS_ORG = MinMax(MS_ORG);
MS_ORG4=imresize(MS_ORG,4,"nearest");

load Datasets1/WorldView3T.mat
MS_ORG = double(MS2);
MS_ORG = MinMax(MS_ORG);
MS_ORG5=imresize(MS_ORG,2,"nearest");
%% data patching
imSz = size(MS_ORG);
patchSz = [16 16 8];
MSpatches1 = dataPatcher1(MS_ORG1, patchSz);
MSpatches2 = dataPatcher1(MS_ORG2, patchSz);
MSpatches3 = dataPatcher1(MS_ORG3, patchSz);
MSpatches4 = dataPatcher1(MS_ORG4, patchSz);
MSpatches5 = dataPatcher1(MS_ORG5, patchSz);
MSpatches = cat(4, MSpatches1, MSpatches2, MSpatches3, MSpatches4, MSpatches5);
%% Datastores 

dsTrain = arrayDatastore(MSpatches, IterationDimension=4);
%% Encoder Network
depth = 3;
netE = model(depth);
%analyzeNetwork(netE)
%% Decoder Network
projectionSize = [2 2 128];
numInputChannels = imSz(3);

layersD = [
    featureInputLayer(512)
    projectAndReshapeLayer(projectionSize)
    transposedConv2dLayer(3,64,Cropping="same",Stride=2)
    reluLayer
    transposedConv2dLayer(3,64,Cropping="same",Stride=2)
    reluLayer
    transposedConv2dLayer(3,32,Cropping="same",Stride=2)
    reluLayer
    transposedConv2dLayer(3,32,Cropping="same",Stride=1)
    reluLayer
    transposedConv2dLayer(3,numInputChannels,Cropping="same")
    reluLayer];
%%
netD = dlnetwork(layersD);
%analyzeNetwork(netD)
%% Train Hyperparameters
numEpochs = 80;
miniBatchSize = 2048;
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
numObservationsTrain = size(MSpatches,4);
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
        [loss,gradientsE,gradientsD] = dlfeval(@mseLoss,netE,netD,X);

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

clearvars -except netE
%% WorldView3 Data Set
load Datasets1/WV3_14.mat
MS_ORG=imresize(MS_ORG,4,"nearest");
GTM =imread('Datasets1/WV3_36GTM.bmp');
RGB =imread('Datasets1/WV3_14HVS.png');
PAN =imread('Datasets1/WV3_14PAN.jpg');

% load Datasets1/WV3_00.mat
% MS_ORG=imresize(MS_ORG,4,"nearest");
% GTM =imread('Datasets1/WV3_00HVSGTM.bmp');
% RGB =imread('Datasets1/WV3_00HVS.png');
% PAN =imread('Datasets1/WV3_00PAN.png');

% load Datasets1/WorldView3T.mat
% MS_ORG=MS2;
% GTM =imread('Datasets1/WorldView3T_GTM19.bmp');
% RGB =imread('Datasets1/WorldView3T_RGB.bmp');
% PAN =imread('Datasets1/WorldView3T_PAN.bmp');


figure;
imshow( RGB);

[Nr, Nc, nb] = size(PAN_ORG);
[Nr, Nc, Nb] = size(MS_ORG);
Spectral=uint16(reshape(MS_ORG,[Nr*Nc Nb]));
%% data patching
patchSize=16; % patch size
maxEpochs=40;
PANpatches=dataPatcher2(MS_ORG,patchSize);
%% create train and test & validation datastores
Train_percent=0.8; % percent of total data select randomly for train data
Label=reshape(GTM,[Nr*Nc,1]); % spectral labels
    [trainInds,trainMap]= Train_Sampels(Spectral,Label,Train_percent);
        num_class = max(trainMap,[],"all");
        numTrain = nnz(trainMap);
        trainMap=reshape(trainMap,Nr,Nc);
            [trainInd,valInd,~] = dividerand(numTrain,0.90,0.10,0);
labelTrain  = zeros(numTrain,1);
trainDataPAN= zeros(patchSize,patchSize,Nb,numTrain);
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
    PANpatches = reshape(PANpatches,patchSize,patchSize,Nb,[]);
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
New_model = netE;

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
New_model = connectLayers(New_model,"encoder_out","concat_Spa&Sec/in1");
New_model = connectLayers(New_model,"relu_Spec_128","concat_Spa&Sec/in2");

lgraph = layerGraph(New_model);
analyzeNetwork(lgraph);
%%
options = trainingOptions('adam', ...
    LearnRateSchedule = "piecewise", ...
    Shuffle ="every-epoch",...
    MaxEpochs = maxEpochs,...
    MiniBatchSize = 1024,...
    ValidationFrequency = 50, ...
    ValidationData = dsValSS, ...
    Verbose=0, Plots = 'training-progress');


ctnet3 = trainNetwork(dsTrainSpaSpc,lgraph,options);
Res = uint8(classify(ctnet3,SpatialSpectralF));
Prdictmap=reshape(Labels_predict,[num_r num_c]);
figure;
imagesc(Prdictmap)




clc
clear all
%%
XTrain = digitTrain4DArrayData;
XTest = digitTest4DArrayData;

%%
numLatentChannels = 1024;
imageSize = [28 28 1];

layersE = [
    imageInputLayer(imageSize,Normalization="none")
    convolution2dLayer(3,32,Padding="same",Stride=2)
    reluLayer
    convolution2dLayer(3,64,Padding="same",Stride=2)
    reluLayer
    fullyConnectedLayer(2*numLatentChannels)
    samplingLayer];
%%
projectionSize = [7 7 64];
numInputChannels = imageSize(3);

layersD = [
    featureInputLayer(numLatentChannels)
    projectAndReshapeLayer(projectionSize)
    transposedConv2dLayer(3,64,Cropping="same",Stride=2)
    reluLayer
    transposedConv2dLayer(3,32,Cropping="same",Stride=2)
    reluLayer
    transposedConv2dLayer(3,numInputChannels,Cropping="same")
    sigmoidLayer];
%%
netE = dlnetwork(layersE);
netD = dlnetwork(layersD);
analyzeNetwork(netD);
analyzeNetwork(netE);
%%
numEpochs = 10;
miniBatchSize = 128;
learnRate = 1e-3;
%%
dsTrain = arrayDatastore(XTrain,IterationDimension=4);
numOutputs = 1;

mbq = minibatchqueue(dsTrain,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB", ...
    PartialMiniBatch="discard");
%%
trailingAvgE = [];
trailingAvgSqE = [];
trailingAvgD = [];
trailingAvgSqD = [];
%%
numObservationsTrain = size(XTrain,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
%%
monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info="Epoch", ...
    XLabel="Iteration");
%%
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
%%
dsTest = arrayDatastore(XTest,IterationDimension=4);
numOutputs = 1;

mbqTest = minibatchqueue(dsTest,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");
%%
YTest = modelPredictions(netE,netD,mbqTest);
%%
err = mean((XTest-YTest).^2,[1 2 3]);
figure
histogram(err)
xlabel("Error")
ylabel("Frequency")
title("Test Data")
%%
numImages = 64;

ZNew = randn(numLatentChannels,numImages);
ZNew = dlarray(ZNew,"CB");

YNew = predict(netD,ZNew);
YNew = extractdata(YNew);
%%
figure
I = imtile(YNew);
imshow(I)
title("Generated Images")
%%
New_model = removeLayers(netE, 'layer');

tempLayer = [flattenLayer("Name","flatten_autoenc")];
New_model = addLayers(New_model,tempLayer);
New_model = connectLayers(New_model,"fc","flatten_autoenc");

tempLayers = [
    fullyConnectedLayer(128,"Name","fcc_3")
    batchNormalizationLayer("Name","batchnormm_3")
    reluLayer("Name","reluu_3")
    fullyConnectedLayer(128,"Name","fcc_4")
    batchNormalizationLayer("Name","batchnormm_4")
    reluLayer("Name","reluu_4")
    fullyConnectedLayer(10,"Name","fcc_5")
    batchNormalizationLayer("Name","batchnormm_5")
    reluLayer("Name","reluu_5")
    softmaxLayer("Name","sofmaxx")
    classificationLayer("Name","classoutputt")];

New_model = addLayers(New_model,tempLayers);
New_model = connectLayers(New_model,"flatten_autoenc", "fcc_3");

lgraph = layerGraph(New_model);
analyzeNetwork(lgraph);

%% Load the dataset
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

numTrainFiles = 100;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
[imdsTest,~] = splitEachLabel(imds,numTrainFiles,'randomize');

%%   Options and train

options = trainingOptions('adam', ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.2, ...
    LearnRateDropPeriod=5, ...
    Shuffle="every-epoch",...
    MaxEpochs=10,...
    MiniBatchSize = 32,...
    ValidationFrequency=50, ...
    ValidationData = imdsValidation, ...
    Verbose=0,Plots= 'training-progress');

trained = trainNetwork(imdsTrain,layers,options);

%%  Prediction and confusion matrix

YPred = classify(trained,imdsTest);
YTest = imdsTest.Labels;

figure;
plotconfusion(YTest,YPred)


%%
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
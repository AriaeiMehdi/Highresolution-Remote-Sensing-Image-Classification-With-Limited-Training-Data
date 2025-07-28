clc
clear all

%% WorldView3 Data Set
load Datasets/WV3_36.mat
% MS_ORG = sharpencnmf(MS_ORG,PAN_ORG);
% GTM =imread('Datasets/WV3_36GTM.bmp');
RGB =imread('Datasets/WV3_36HVS.png');

% Scale PAN and pansharpend image between 0-1
PAN_ORG = double(PAN_ORG);
PAN_ORG = MinMax(PAN_ORG);
MS_ORG = double(MS_ORG);
MS_ORG = MinMax(MS_ORG);

Nb = size(PAN_ORG,3);
% Plot RGB and PAN image 
figure;
subplot(1,2,1);
imshow(RGB);
title("RGB Image")
subplot(1,2,2); 
imshow(PAN_ORG);
title("PAN Image")

clearvars -except PAN_ORG MS_ORG Nb
%% data patching

imSz = size(PAN_ORG);
patchSz = [16 16];
xIdxs = [1:patchSz(2):imSz(2) imSz(2)+1];
yIdxs = [1:patchSz(1):imSz(1) imSz(1)+1];
patches = cell(length(yIdxs)-1,length(xIdxs)-1);
for i = 1:length(yIdxs)-1
    Isub = PAN_ORG(yIdxs(i):yIdxs(i+1)-1,:);
    for j = 1:length(xIdxs)-1
        patches{i,j} = Isub(:,xIdxs(j):xIdxs(j+1)-1);
    end
end
PANpatches = reshape( cat(3,patches{:}), [16,16,1,size(patches,1)^2]);
dsTrain = arrayDatastore(PANpatches, IterationDimension=4);
% [~,~,~,n] = size(PANpatches) ;
% P = 0.80 ;
% idx = randperm(n)  ;
% PANpatches_Training = PANpatches(:, :, :, idx(1:round(P*n))); 
% PANpatches_Testing = PANpatches(:, :, :, idx(round(P*n)+1:end));
% 
% dsTrain = arrayDatastore(PANpatches_Training, IterationDimension=4);
% dsTest = arrayDatastore(PANpatches_Testing, IterationDimension=4);

clearvars -except PANpatches_Testing PANpatches_Training PANpatches PAN_ORG MS_ORG dsTrain dsTest Nb

%% Model
ModelInputSize = [16 16 Nb];
depth = 2;
Latentdepth = 256;
Autoencoder_model = AutoEnc_model(ModelInputSize, depth, Latentdepth);

clearvars -except PANpatches_Testing PANpatches_Training PANpatches PAN_ORG MS_ORG Autoencoder_model dsTrain dsTest ModelInputSize Nb

%% Train Hyperparameters
numEpochs = 150;
miniBatchSize = 512;
learnRate = 1e-6;
%%  prepare mini-batch for train
numOutputs = 1;

mbq = minibatchqueue(dsTrain,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFormat="SSCB", ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    PartialMiniBatch="discard");

%% Initialize train parameters
vel = [];
%%
numObservationsTrain = size(PANpatches,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
%%
monitor = trainingProgressMonitor(Metrics="Loss");
monitor.Info = ["LearningRate","Epoch","Iteration","ExecutionEnvironment"];
monitor.XLabel = "Iteration";
monitor.Status = "Training";
monitor.Progress = 0;
executionEnvironment = "auto";

updateInfo(monitor,LearningRate=learnRate);

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    updateInfo(monitor,ExecutionEnvironment="GPU");
else
    updateInfo(monitor,ExecutionEnvironment="CPU");
end

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

        % If training on a GPU, then convert data to a gpuArray.
        if canUseGPU
            X = gpuArray(X);
        end

        % Evaluate loss and gradients.
        [loss, gradients] = dlfeval(@mseLoss,Autoencoder_model,X);

        % Update the network parameters using the SGDM optimizer.
        [net,vel] = sgdmupdate(Autoencoder_model,gradients,vel);

        % Update the training progress monitor. 
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100*iteration/numIterations;
    end
end

%% Visualize autoencoder output
shuffle(mbq);
input_batch = next(mbq);
output_batch = forward(Autoencoder_model, input_batch);
output_batch = extractdata(output_batch(:,:,:,1:2));
input_batch = extractdata(input_batch(:,:,:,1:2));

% Plot input image to model 
figure;
subplot(1,2,1);
imshow(reshape(input_batch(:,:,:,1), ModelInputSize));
title("Image 1")
subplot(1,2,2); 
imshow(reshape(input_batch(:,:,:,2), ModelInputSize));
title("Image 2")


% Plot model output
figure;
subplot(1,2,1);
imshow(reshape(output_batch(:,:,:,1), ModelInputSize));
title("Output 1")
subplot(1,2,2); 
imshow(reshape(output_batch(:,:,:,2), ModelInputSize));
title("Output 2")


%% detach encoder part

alllayers = Autoencoder_model.Layers; 

for i = 1:numel(alllayers)
    if alllayers(i,1).Name == "LatentNetworkLayer9"
        index = i;
    end
end

redundant_layers = {alllayers((index+1):end,1).Name};
Encoder_net = removeLayers(Autoencoder_model, redundant_layers);
Encoder_net = initialize(Encoder_net);

%% data patching
patchSize=16; % patch size
PANpatches=dataPatcher2(PAN_ORG,patchSize);     % The data is in this order (spatial, spatial, channel, Y, X)
num_patches = size(PANpatches,4)*size(PANpatches,5);
PANpatches = reshape(PANpatches, [16 16 1 num_patches]);    % It reshapes data columnwise
%%  prepare mini-batch for test output
dsImage = arrayDatastore(PANpatches, IterationDimension=4);
numOutputs = 1;
miniBatchSize = 128;

minibatchque = minibatchqueue(dsImage,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFormat="SSCB", ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    PartialMiniBatch="discard",...
    OutputEnvironment = "auto");

output_b = [];
while hasdata(minibatchque)
    data = next(minibatchque);
    output = forward(Encoder_net, data);
    output_b = [output_b output];
end

output_b = extractdata(output_b)';
x = reshape(output_b, [1024 1024 256]);
L = imsegkmeans(x,10);
imagesc(L);


%% WorldView3 Data Set
load Datasets/WV3_36.mat
MS_ORG=double(MS_ORG);
MS_ORG=imresize(MS_ORG,4,"nearest");
% GTM =imread('data/WV3MS_GTMH2.bmp');% imread('data/WV3_GTM1.bmp');
GTM =imread('Datasets/WV3_36GTM.bmp');% imread('data/WV3_GTM1.bmp');
labelClass=["Concrete","Boat","Car","Tree","Water","Asphalt","Sea","Building","Shelter","Unknown","Soil"];
    RGB =imread('Datasets/WV3_36HVS.png');
    PAN =imread('Datasets/WV3_36PAN.png');
    figure;imshow( RGB);
    % RGB(:,:,1)=31*GTM(:,:);
% GTM=imresize(GTM,0.25,"nearest");
% RGB = MS_ORG(:,:,[2 3 5]);RGB = MS_ORG(:,:,[2 3 5]);RGB = MUL(:,:,[2 3 5]);
% PAN_ORG=imresize(PAN_ORG,0.25,"nearest");%PAN1;%
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

ctnet3 = trainNetwork(dsTrainSpaSpc,lgraph,options);
Res = uint8(classify(ctnet3,SpatialSpectralF));    % save('cessut3.mat','ctnet3');
Classification_Performance(uint8(Label),Res,GTM,PAN,RGB,"DCNN Spatial-Spectral Classification",labelClass);







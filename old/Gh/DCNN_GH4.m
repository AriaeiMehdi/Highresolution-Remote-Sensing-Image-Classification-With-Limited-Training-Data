%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image processing and Information Analysis Lab.
% Faculty of Electrical and Computer Engineering, 
% Tarbiat Modares University (TMU), Tehran, Iran;
% DCNN Image Spatial Feature Extraction and Classification 
% By Hassan Ghassemian Ver. 2023.5.21
%   gpuDeviceTable
%   gpuDevice(1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc;
clear;
clear all;
 % close all;
  % warning off;
%% load dataset for MS or HSI input
% load Datasets\Indian_pines_corrected.mat;    % load data
% load Datasets\Indian_pines_gt.mat;           % load GTM%[GTM,cmap] = imread('Datasets/PaviaU_gt.png');%Datasets/PaviaU.mat
% GTM=uint8(indian_pines_gt);
% MS_ORG = indian_pines_corrected;%cell2mat(struct2cell(paviaU)); % just for .mat files (convert struct to matrix)
% % im_HS=load("Salinas.mat").salinas;
% % gt=load("Salinas_gt.mat").salinas_gt;
% clear indian_pines_corrected indian_pines_gt
%% PCA & PAN & RCB
% [Nr, Nc, Nb] = size(MS_ORG);
% Spectral= reshape(MS_ORG,[Nr*Nc Nb]);
% [W,PCAData,Landas,tsquared,Normal_Landas,mu] = pca(Spectral);
% nb=1;
% NPCA=30;
% c=double(2^16-1);
% for i=1:NPCA
%     a=(PCAData(:,i)-min(PCAData(:,i)))/(max(PCAData(:,i))-min(PCAData(:,i)));
%     SpectralPCA(:,i)=c*a;
% end
%  % SpectralPCA=double(SpectralPCA);
% for i=1:3
%     a=(PCAData(:,i)-min(PCAData(:,i)))/(max(PCAData(:,i))-min(PCAData(:,i)));
%     RGB(:,i)=uint8(round(255*a));
% end
% RGB=reshape(RGB,[Nr Nc 3]);%imshow(RGB);
% for i=1:nb
%     PAN_ORG(:,i)=SpectralPCA(:,i);%RGB;%
% end
% PAN_ORG=reshape(PAN_ORG,[Nr Nc nb]);
% PAN=uint8((double(PAN_ORG(:,:,1)))/256); %imshow(PAN);
% clear a c W Landas tsquared mu Normal_Landas PCAData im_PCs;
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
tStart = tic;
patchSize=16; % patch size
maxEpochs=2;
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
 tEnd = toc(tStart);
%  % Dataset and trainset information 
%      label2 = categorical(Label);
%      dsTabel =    arrayDatastore(label2);
% Dataset = combine(SpatialSpectralF,dsTabel);
% % spltp = splitlabels(SpatialSpectralF,[0.85 0.12]);
% Dataset_info = countlabels(Dataset,'UnderlyingDatastoreIndex',3);
% dsTrain_info = countlabels(dsTrainSpaSpc,'UnderlyingDatastoreIndex',3);
% % dsVal_info = countlabels(dsValSS,'UnderlyingDatastoreIndex',3);
% fig1 = uifigure(Name="Dataset Info");
% uitable(fig1,"Data",Dataset_info, "Position",[20 20 500 350]);
% fig2 = uifigure(Name="TrainSet Info");
% uitable(fig2,"Data",dsTrain_info, "Position",[20 20 500 350]);
% fig3 = uifigure(Name="TrainSet Info");
% uitable(fig3,"Data",dsVal_info, "Position",[20 20 500 350]);
    %% SVM_Classification
% tic
% [Predict_SVM] = SVM_Classifier(SpectralPCA,Label,SpectralPCA(trainInds, :),Label(trainInds));
% Classification_Performance(uint8(Label),uint8(Predict_SVM),GTM,PAN,RGB,"SVM Classifier PCA30 Bd");
    % tic
    % [Predict_SVM] = SVM_Classifier(double(Spectral),Label,double(Spectral(trainInds, :)),Label(trainInds));
    % Classification_Performance(uint8(Label),uint8(Predict_SVM),GTM,PAN,RGB,"SVM Classifier 200 Bd");
%% spatial-spectral network
tStart3 = tic;
%     output3=mynet3(patchSize,Nb,[Nr,Nc],num_class,dsTrainSpaSpc,dsValSS,SpatialSpectralF,nb);
%       Res = mynet3(patchSize,Nb,[Nr,Nc],num_class,dsTrainSpaSpc,dsValSS,SpatialSpectralF,nb)
    % layer of networks
    lgraph = layerGraph();
    tempLayers = [
% input data and create convolution tokens
        imageInputLayer([patchSize patchSize nb],"Name","imagein_A")
        convolution2dLayer([5 5],16,"Name","conv_A","Padding","same")
        batchNormalizationLayer("Name","batchNorm_A")
        reluLayer("Name","relu_A")
            convolution2dLayer([7 7],16,"Name","conv_A1","Padding","same")
            batchNormalizationLayer("Name","batchNorm_A1")
            reluLayer("Name","relu_A1")
            % maxPooling2dLayer(2,'Stride',2)
                % convolution2dLayer([9 9],16,"Name","conv_A2","Padding","same")
                % batchNormalizationLayer("Name","batchNorm_A2")
                % reluLayer("Name","relu_A2")
                % maxPooling2dLayer(2,'Stride',2)
                    % convolution2dLayer([11 11],16,"Name","conv_A3","Padding","same")
                    % batchNormalizationLayer("Name","batchNorm_A3")
                    % reluLayer("Name","relu_A3")
                    % % maxPooling2dLayer(2,'Stride',2)
                    %     convolution2dLayer([13 13],16,"Name","conv_A4","Padding","same")
                    %     batchNormalizationLayer("Name","batchNorm_A4")
                    %     reluLayer("Name","relu_A4")
                        globalMaxPooling2dLayer("Name","pool_A1&A5")
        % averagePooling2dLayer(poolSize,'Stride',strideSize,"Name","pool_A1&A5")
        flattenLayer("Name","flatten_Spa")
        reluLayer("Name","relu_Spa_64_128")
        fullyConnectedLayer(128,"Name","fullyConct_64_128")
        reluLayer("Name","relu_Spa_128")];
                        % maxPooling2dLayer(2,'Stride',2)
    lgraph = addLayers(lgraph,tempLayers);
    tempLayers = [
        imageInputLayer([1 1 Nb],"Name","Spectral_Nb")
        flattenLayer("Name","flatten")
        fullyConnectedLayer(128,"Name","fullyConct_8_128")
        reluLayer("Name","relu_Spec_8_128");
            fullyConnectedLayer(128,"Name","fullyConct_128_128")
            reluLayer("Name","relu_Spec_128_128");
                fullyConnectedLayer(128,"Name","fullyConct_128")
                reluLayer("Name","relu_Spec_128")];
    % lgraph = addLayers(lgraph,tempLayers);
    % tempLayers = [
    %     concatenationLayer(3,2,"Name","concat_A1&A5")
    %     % globalAveragePooling2dLayer("Name","pool_A1&A5")
    %     globalMaxPooling2dLayer("Name","pool_A1&A5")
    %     % averagePooling2dLayer(poolSize,'Stride',strideSize,"Name","pool_A1&A5")
    %     flattenLayer("Name","flatten_Spa")
    %     reluLayer("Name","relu_Spa_128")];
    lgraph = addLayers(lgraph,tempLayers);
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
    lgraph = addLayers(lgraph,tempLayers);
    % clean up helper variable
    clear tempLayers;
    % lgraph = connectLayers(lgraph,"relu_A","concat_A1&A5/in1");
    % lgraph = connectLayers(lgraph,"relu_A4","concat_A1&A5/in2");
    lgraph = connectLayers(lgraph,"relu_Spa_128","concat_Spa&Sec/in1");
    lgraph = connectLayers(lgraph,"relu_Spec_128","concat_Spa&Sec/in2");
    
    analyzeNetwork(lgraph);  % figure; plot(lgraph)
    % set option parameters for training process (epochs-validation samples...)
       options = trainingOptions('adam', ...
        LearnRateSchedule = "piecewise", ...
        Shuffle ="every-epoch",...
        MaxEpochs = maxEpochs,...
        MiniBatchSize = 128,...
        ValidationFrequency = 50, ...
        ValidationData = dsValSS, ...
        Verbose=0, Plots = 'training-progress');
    ctnet3 = trainNetwork(dsTrainSpaSpc,lgraph,options);
    Res = uint8(classify(ctnet3,SpatialSpectralF));    % save('cessut3.mat','ctnet3');
    Classification_Performance(uint8(Label),Res,GTM,PAN,RGB,"DCNN Spatial-Spectral Classification",labelClass);
 tEnd = toc(tStart3)
 GTM2=Res;
 Num_test = size(Res,1);
for ts=1:Num_test
    if GTM2(ts)~=Label(ts)
            GTM2(ts)=0;
    end
end
GTM2 = uint8(reshape(GTM2,[Nr Nc 1]));
kk2=double(max(GTM,[],"all"));
cmap=jet(kk2+1); cmap(1, :)=[1.0 1.0 1.0];
% figure;imshow( GTM2,[]);colorbar;colormap(cmap);
imwrite(GTM2,cmap,'GTM25_30new4.bmp');
cmap=my_map(kk2); 
Prdictmap=reshape(Res,[Nr Nc]);
figure; imshow( Prdictmap ,cmap);title('Total Class Map');colorbar('Ticks',[1,2,3,4,5,6,7,8,9,10,11],...
    'TickLabels',{"Concrete","Boat","Car","Tree","Water","Asphalt","Sea","Building","Shelter","Unknown","Soil"});
newmap=[1 1 1; 0.5 0.5 0.5;1 0 0; 0.5 0 0; 0 1 0;0 1 1;0 0 0; 0 0 1; 1 1 0;0.25 0.25 0.25; 0.75 0 0.75; 1 0.5 0];
figure; imshow( Prdictmap ,newmap);title('Total Class Map');colorbar('Ticks',[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5],...
    'TickLabels',{"Unknown","Concrete","Boat","Car","Tree","Water","Asphalt","Sea","Building","Shelter","Shadow","Soil"});
    % 0 0.5 0;
    % 1 0 0;
    % 0.5 0.2 0.5;
    % 0 1 0;
    % 0.5 0 0;
    % 0 0 1;
    % 0.75 0.25 0;
    % 1 1 0;
    % 0 0 0.5;
    % 1 0 1;...
    % 0.75 0 0.75;
    % 0 1 1;
    % 0.5 0.5 0.5;
    % 1 0.5 0;
    % 0 0 0];
% ["Concrete","Boat","Car","Tree","Water","Asphalt","Sea","Building","Shelter","Unknown","Soil"]
% figure;
% kk=0;
% for i=1 :64
%     for j=1:8
%         fff=ctnet3.Layers(14, 1).Weights(:,:,i,j);
%             filters=imresize(fff  ,32,"bilinear");
%             kk=kk+1;
%             subplot(16,32,kk);
%             imshow( filters ,[]);
%     end
% end
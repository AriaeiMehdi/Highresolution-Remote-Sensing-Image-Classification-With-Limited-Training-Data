load Datasets1/WV3_15.mat
MS_ORG=imresize(MS_ORG,4,"nearest");
RGB =imread('Datasets1/WV3_15HVS.png');
PAN =imread('Datasets1/WV3_15PAN.jpg');

figure;
imshow( RGB);

[Nr, Nc, nb] = size(PAN_ORG);
[Nr, Nc, Nb] = size(MS_ORG);
Spectral=uint16(reshape(MS_ORG,[Nr*Nc Nb]));
%%
patchSize=16; % patch size
maxEpochs=10;
PANpatches=dataPatcher2(MS_ORG,patchSize);
%% create train and test & validation datastores
PANpatches = reshape(PANpatches,patchSize,patchSize,Nb,[]);
spec=zeros(1,1,Nb,Nr*Nc);
spec(1,1,:,:)=Spectral.';
%% Datastores 
SpatialFeatures = arrayDatastore(PANpatches,IterationDimension=4);
SpectralFeatures= arrayDatastore(spec,IterationDimension=4);
SpatialSpectralF = combine(SpatialFeatures,SpectralFeatures);
%%
Res = uint8(classify(ctnet3,SpatialSpectralF));
Prdictmap=reshape(Res,[1024 1024]);
imagesc(Prdictmap);
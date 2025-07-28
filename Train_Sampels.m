%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image processing and Information Analysis Lab.
% Faculty of Electrical and Computer Engineering, Tarbiat Modares University (TMU), Tehran, Iran;
% Random Training Samples Generation by Hassan Ghassemian Ver. 2013.07.28
%
% Label:    True Label (Class) of Feature Vectors (Label=0 Unknown Class)
% Train_percent:            Percentage of total training samples for each class
% minNumofTrainingSamples:  Minimum number of training samples for each class
% maxNumofTrainingSamples:  Maximum number of training samples for each class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[trainInds,trainMap] = Train_Sampels(Feature,Label,Train_percent)
random=1;
% random:   If random = 1   Training samples are selected randomly (i.e, distributed all over the class)
%           If random = 0   Training samples are selected as a whole piece
%%
[Total_Train_Number,Num_Features]=size(Feature);
classLabels = unique(Label(Label ~= 0));
Num_Classes = length(classLabels);
minNumofTrainingSamples=min(2*Num_Features,50);             %Min should be adjusted
maxNumofTrainingSamples=min(fix(Total_Train_Number/Num_Classes),1000);
Train_Number(1:Num_Classes) = Train_percent;
Test_Number = zeros( 1,Num_Classes );
trainMap = zeros(size(Label));
for nc = 1:Num_Classes
    inds = find(Label == classLabels(nc));
    N_Total = numel(inds); % number of pixels in the class
    Test_Number(nc) =N_Total;
    if Train_Number(nc) < 1 % Nt is the percentage
        Train_Number(nc) = fix(Train_Number(nc) * N_Total);
    end
    if Train_Number(nc) < minNumofTrainingSamples
        Train_Number(nc) = minNumofTrainingSamples;
    end
    if Train_Number(nc) > maxNumofTrainingSamples
        Train_Number(nc) = maxNumofTrainingSamples;
    end
    if random
        inds = inds(randperm(length(inds)));
    end
    trainMap(inds(1:min(numel(inds),Train_Number(nc)))) = classLabels(nc);
end
trainInds = find(trainMap ~= 0);
% Label_train=Label(trainInds);
% Feature_train=Feature(trainInds, :);
% Label_test=Label(Label~=0,:);
% Feature_test=Feature(Label~=0,:);
%%
% LabeTrain=size(Label_train)
% FeaTrain=size(Feature_train)
% LabeTest=size(Label_test)
% FeaTest=size(Feature_test)
% Total_Train_Number
Test_Number
Train_Number
end


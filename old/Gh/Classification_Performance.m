%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image processing and Information Analysis Lab.
% Faculty of Electrical and Computer Engineering, Tarbiat Modares University (TMU), Tehran, Iran;
% Classification Performance by Hassan Ghassemian Ver. 2019.05.08
%
% Labels:   True Label (Class) of Feature Vectors (Label=0 Unknown Class)
% Labels_predict:   Predicted Class of FeatureVectors
% im:               Pan Image
% classmember:      Normalized Class Membership or Class Belongness Number
% str:              Name of Classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[] = Classification_Performance(Labels,Labels_predict,GTM,PAN,RGB,str,labelClass)
time_Classification=round(toc);
tic
[num_r,num_c,k]=size(GTM);
Num_classes = nnz(unique(Labels));
AA=size(unique(Labels));
Num_Labels = AA(1);
% Labels=uint16(Labels);
% Labels_predict=uint16(Labels_predict);     
Prdictmap=reshape(Labels_predict,[num_r num_c]);
MAP =Prdictmap;
%% Color Map
mymap=jet(Num_classes+1);
mymap(1, :)=[1.0 1.0 1.0]; %[0.0 0.0 0.0];
 if (Num_classes ~= Num_Labels)  
     for i=1:num_r
        for j=1:num_c
            if GTM(i,j) == 0 
                MAP(i,j)= 0;
            end
        end
      end
 end 
%% Overall and Average Accuracy
 confusion1 = confusionmat(Labels,Labels_predict);
if (Num_classes ~= Num_Labels) % remove class zero
    for i=1:Num_classes
        for j=1:Num_classes
            confusion(i,j)=confusion1(i+1,j+1);
        end
    end
else
        confusion=confusion1;
end   
Acc_C_ML=zeros(Num_classes,1);
for i=1:Num_classes
    Acc_C_ML(i,1)=confusion(i,i)/ sum(confusion(i,:));
end
AA_ML =sum( Acc_C_ML)/Num_classes;
AA=round(100.*AA_ML);
OA_ML = sum(diag(confusion))/sum(sum(confusion));
OA=round(100.*OA_ML);
 %% Overall and Average Validity
Val_C_ML=zeros(1,Num_classes);
for i=1:Num_classes
    Val_C_ML(1,i)=(confusion(i,i)/ sum(confusion(:,i)))';
end
AV_ML = (sum( Val_C_ML))/Num_classes; AV=round(100.*AV_ML);
Pe_t_ML =0;
for i=1:Num_classes
Pe_C_ML=(sum(confusion(:,i))*sum(confusion(i,:)));
Pe_t_ML=Pe_C_ML+Pe_t_ML;
end
Pe_ML=Pe_t_ML/(sum(sum(confusion)).^2);
Kappa_ML =(OA_ML-Pe_ML)/(1-Pe_ML);
Kappa =round(100.*Kappa_ML);
figure
        sgt= sgtitle({str; '  '}, 'Color','red');
        sgt.FontSize = 20;
        cm=confusionchart(confusion,labelClass, ...
        'Title',['Overall Accuracy=',num2str(OA),'   Average Accuracy=',num2str(AA),'   Average Validity=',num2str(AV), ...
        '   Kappa=',num2str(Kappa),'   Classification time=',num2str(time_Classification),'  Seconds'], ...
        'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
% figure
% sgt= sgtitle({str; '  '}, 'Color','red');
% sgt.FontSize = 20;
% cm=confusionchart(confusion, ...
%     'Title',['Overall Accuracy=',num2str(OA),'   Average Accuracy=',num2str(AA),'   Average Validity=',num2str(AV), ...
%     '   \kappa=',num2str(Kappa),'   Classification time=',num2str(time_Classification),'  Seconds'], ...
%     'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
%         title('The Original Image')
%            sgtitle ({ ['Overall Accuracy = ',num2str(OA),...
%         '  Average Accuracy = ',num2str(AA),'  Average Validity = ',...
%         num2str(AV),'  \kappa = ',num2str(Kappa),...
%         '  Classification time = ',num2str(time_Classification),...
%         ' Seconds'];' '});
%%
    for i=1:Num_classes
        confusion(i,:)=round(100*(confusion(i,:)/ sum(confusion(i,:))));
    end
    % figure
    %     cm = confusionchart(confusion,'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
% %% Overall and Average Accuracy
% % confusion=confusion1;
% Acc_C_ML=zeros(Num_classes,1);
% for i=1:Num_classes
%     Acc_C_ML(i,1)=confusion(i,i)/ sum(confusion(i,:));
% end
% AA_ML =sum( Acc_C_ML)/Num_classes;
% AA=round(100.*AA_ML);
% OA_ML = sum(diag(confusion))/sum(sum(confusion));
% OA=round(100.*OA_ML);
%  %% Overall and Average Validity
%     Val_C_ML=zeros(1,Num_classes);
%     for i=1:Num_classes
%         Val_C_ML(1,i)=(confusion(i,i)/ sum(confusion(:,i)))';
%     end
% 
%     AV_ML = (sum( Val_C_ML))/Num_classes; AV=round(100.*AV_ML);
%     Pe_t_ML =0;
%     for i=1:Num_classes
%         Pe_C_ML=(sum(confusion(:,i))*sum(confusion(i,:)));
%         Pe_t_ML=Pe_C_ML+Pe_t_ML;
%     end
% Pe_ML=Pe_t_ML/(sum(sum(confusion)).^2);
% Kappa_ML =(OA_ML-Pe_ML)/(1-Pe_ML);
% Kappa =round(100.*Kappa_ML);
%     figure
%         sgt= sgtitle({str; '  '}, 'Color','red');
%         sgt.FontSize = 20;
%         cm=confusionchart(confusion, ...
%         'Title',['Overall Accuracy=',num2str(OA),'   Average Accuracy=',num2str(AA),'   Average Validity=',num2str(AV), ...
%         '   Kappa=',num2str(Kappa),'   Classification time=',num2str(time_Classification),'  Seconds'], ...
%         'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
%% Comparing Class Mab with The Reference Map 
Prdictmap(1,1)=0;
MAP(1,1)=0;    %assign at least one pixel to background (class 0)
GTM(1,1)=0;
figure,
    subplot(2,3,1);
        imshow( GTM,[]);
        colorbar('TickLabels',{"Unknown","Concrete","Boat","Car","Tree","Water","Asphalt","Sea","Building","Shelter","Shadow","Soil"});
        title('Reference Map');
        colormap(mymap); 
    subplot(2,3,2);
        imshow( MAP ,[]);
        colorbar('TickLabels',{"Unknown","Concrete","Boat","Car","Tree","Water","Asphalt","Sea","Building","Shelter","Shadow","Soil"});
        title('Test Class Map');
        colormap(mymap);
    subplot(2,3,3);
        imshow( (Prdictmap) ,[]);
        colorbar('TickLabels',{"Unknown","Concrete","Boat","Car","Tree","Water","Asphalt","Sea","Building","Shelter","Shadow","Soil"});
        title('Total Class Map');
        colormap(mymap);
    subplot(2,3,4);
        imshow(RGB,[]);
        title('The RGB Image')
    subplot(2,3,5);
        imshow(PAN,[]);
        title('The PAN Image')
               sgtitle ({['Overall Accuracy = ',num2str(OA),...
            '  Average Accuracy = ',num2str(AA),'  Average Validity = ',...
            num2str(AV),'  \kappa = ',num2str(Kappa),...
            '  Classification time = ',num2str(time_Classification),...
            ' Seconds']}); 
    subplot(2,3,6);
        cm = confusionchart(confusion,labelClass);
        title(str)
        % cm=confusionchart(confusion,labelClass, ...
        % 'Title',['Overall Accuracy=',num2str(OA),'   Average Accuracy=',num2str(AA),'   Average Validity=',num2str(AV), ...
        % '   Kappa=',num2str(Kappa),'   Classification time=',num2str(time_Classification),'  Seconds'], ...
        % 'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
% mymap=my_map(11);        
figure; 
imshow( (Prdictmap) ,[]);
colorbar('TickLabels',{"Unknown","Concrete","Boat","Car","Tree","Water","Asphalt","Sea","Building","Shelter","Shadow","Soil"});
title('Total Class Map');
colormap(mymap);
% figure; imshow( MAP ,[]); colorbar;title('Test Class Map');colormap(mymap);

end

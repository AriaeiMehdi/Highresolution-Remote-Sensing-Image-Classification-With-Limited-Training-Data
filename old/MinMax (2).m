%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image processing and Information Analysis Lab.
% Faculty of Electrical and Computer Engineering, 
% Tarbiat Modares University (TMU), Tehran, Iran;
% Scale the input image  
% By Mehdi Ariaei V1.0. 2023.08.08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Image:  The input image to be scaled
%   Mode:   Output data type

%           Options:
%           NaN:    The Image will be scaled between -1 to 1
%           uint8:  The Image will be scaled between -255 to 255
%           uint16: The Imgae will be scaled between -65535 to 65535




function scaled = MinMax(Image, Mode)


    scaled = [];
    [~,~, num_bands] = size(Image);
    for i = 1:num_bands
        min_val = min(Image(:, :, i), [], 'all');
        max_val = max(Image(:, :, i), [], 'all');
    
        % Scale the image to the range [ |Image| < 1 ]
        temp = (double(Image(:, :, i)) - min_val) / (max_val - min_val);
        scaled = cat(3, scaled, temp);
    end 

    
    if (exist('Mode', 'var'))
        if Mode == "uint8"
            scaled = uint8(scaled * 255);
        elseif Mode == "uint16"
            scaled = uint16(scaled * 65535);
        end
    end

end
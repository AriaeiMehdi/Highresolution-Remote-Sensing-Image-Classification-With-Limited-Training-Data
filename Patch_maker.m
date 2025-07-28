%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image processing and Information Analysis Lab.
% Faculty of Electrical and Computer Engineering, 
% Tarbiat Modares University (TMU), Tehran, Iran;
% Make a set of patches from input image  
% By Mehdi Ariaei V1.0. August, 9th, 2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [patches, center_pixel] = Patch_maker(IMG, Patch_Size, Mode)
    
    % Make patch size an odd number if it is even
    if mod(Patch_Size, 2) == 0
        Patch_Size = Patch_Size + 1;
    end

    [height, width, depth] = size(IMG);
    IMG_padded = padarray(IMG,[(Patch_Size-1)/2 (Patch_Size-1)/2],0,'both');  
   
    patches = [];
    center_pixel = [];
    center=(Patch_Size+1)/2;

    for i = 1:height
        for j = 1:width
            if (exist('Mode', 'var'))
                if Mode == "pixel"
                    pixel = IMG(i, j,:);
                    center_pixel = cat(4, center_pixel, pixel);
                    
                elseif Mode == "patch"
                    patch = IMG_padded(i:(i+Patch_Size-1), j:(j+Patch_Size-1), :);
                    patches = cat(4, patches, patch);   
                elseif Mode == "both"  
                    patch = IMG_padded(i:(i+Patch_Size-1), j:(j+Patch_Size-1), :);
                    patches = cat(4, patches, patch);
                    pixel = IMG(i, j,:);
                    center_pixel = cat(4, center_pixel, pixel);
                end
            end
        end
    end
end
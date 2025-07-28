% create spatial patches from hyperspectral image
function patched = dataPatcher2(im,patchSize)
    if mod(patchSize,2)==0
        patched1 = dataPatcher2(im,patchSize+1);
        patched = patched1(2:end,2:end,:,:,:);
    else
        [sx,sy,n_features] = size(im);
        p3 = floor(patchSize/2) + 1;
        p4 = p3-1;
        patched =(ones(patchSize,patchSize,n_features,sx,sy,'double'));
        im = padarray(im,[p4,p4],'symmetric','both');
        i = 0;
        patch = (zeros(patchSize,patchSize,n_features,'double'));
        for xindex = p3:sx+p4
            i = i+1;
            j = 0;
            for yindex = p3:sy+p4
                j = j+1;
                patch(:,:,:) = im(xindex-p4:xindex+p4,yindex-p4:yindex+p4,:);
                patched(:,:,:,i,j) = patch;
            end
        end
    end
end
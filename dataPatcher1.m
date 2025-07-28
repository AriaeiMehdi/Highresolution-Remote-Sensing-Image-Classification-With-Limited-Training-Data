function MSpatches = dataPatcher1(image, patchSz)
    imSz = size(image);
    xIdxs = [1:patchSz(2):imSz(2) imSz(2)+1];
    yIdxs = [1:patchSz(1):imSz(1) imSz(1)+1];
    patches = cell(length(yIdxs)-1,length(xIdxs)-1);
    for i = 1:length(yIdxs)-1
        Isub = image(yIdxs(i):yIdxs(i+1)-1,:,:);
        for j = 1:length(xIdxs)-1
            patches{i,j} = Isub(:,xIdxs(j):xIdxs(j+1)-1, :);
        end
    end
    patches = patches';
    MSpatches = reshape( cat(3,patches{:}), [patchSz(1),patchSz(2),patchSz(3),size(patches,1)^2]);
end


function pca_result = pca_Trans(input_image, PC_num)

    [h, w, bands] = size(input_image);

    % Reshape the data to a 2D matrix
    X = reshape(input_image, h*w, bands);
    
    % Perform mean centering
    X_mean = mean(X);
    X_centered = X - X_mean;
    
    % Compute covariance matrix
    cov_matrix = cov(X_centered);
    
    % Perform eigendecomposition
    [eigenvectors, eigenvalues] = eig(cov_matrix);
    
    % Sort the eigenvectors in descending order of eigenvalues
    [eigenvalues, indices] = sort(diag(eigenvalues), 'descend');
    eigenvectors = eigenvectors(:, indices);
    
    % Select the first k eigenvectors to reduce dimensionality
    components = eigenvectors(:, 1:PC_num);
    
    % Project data onto the principal components
    scores = X_centered * components;
    
    % Reshape the scores back into an image format
    pca_result = reshape(scores, h, w, PC_num);

end
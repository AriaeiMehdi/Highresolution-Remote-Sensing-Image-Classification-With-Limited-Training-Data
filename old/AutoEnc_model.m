function Autoencoder_model  = AutoEnc_model(inputSize, depth, Latentdepth)
    
    divisible =~rem(inputSize(1),2^depth)*(inputSize(1))/(2^depth); 
    if divisible > 1
        projectionSize = [inputSize(1)/2^depth inputSize(2)/2^depth Latentdepth]; 
    else
        msg = 'Network depth and input size are not divisible';
        error(msg)
    end

    encoderBlock = @(block) [
        convolution2dLayer(3,2^(5+block),"Padding",'same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,2^(5+block),"Padding",'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,"Stride",2)];
    encoder = blockedNetwork(encoderBlock,depth,"NamePrefix","encoder_");
    
    decoderBlock = @(block) [
        transposedConv2dLayer(2,2^(10-block),'Stride',2)
        batchNormalizationLayer
        convolution2dLayer(3,2^(10-block),"Padding",'same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,2^(10-block),"Padding",'same')
        batchNormalizationLayer
        reluLayer];
    decoder = blockedNetwork(decoderBlock,depth,"NamePrefix","decoder_");
    
    bridge = [
        convolution2dLayer(3,Latentdepth,"Padding",'same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,Latentdepth,"Padding",'same')
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.5)
        globalAveragePooling2dLayer('Name','gap1')
        flattenLayer
        projectAndReshapeLayer(projectionSize)];   

    Autoencoder_model = encoderDecoderNetwork(inputSize,encoder,decoder, ...
        "OutputChannels",inputSize(3), ...
        "SkipConnections","concatenate", ...
        "LatentNetwork",bridge);

    analyzeNetwork(Autoencoder_model)
    
    
    
    
end






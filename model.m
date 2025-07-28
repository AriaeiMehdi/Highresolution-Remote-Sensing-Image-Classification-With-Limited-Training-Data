function encoder = model(depth)

    encoder = resnet18("Weights","none");
    old_input_layer_name = encoder.Layers(1, 1).Name;
    new_input_layer = [imageInputLayer([16 16 8], "Name",'Input_Image', 'Normalization','none')];
    encoder = replaceLayer(encoder, old_input_layer_name, new_input_layer);
    encoder = removeLayers(encoder, {'ClassificationLayer_predictions', 'prob','fc1000','pool5'});

    for i = 1:numel(encoder.Layers)
        if strcmp(encoder.Layers(i, 1).Name, append('res', int2str(depth), 'b_relu'))
            redundant_layers = {encoder.Layers(i+1:end, 1).Name};
            break;
        end
    end
    encoder = removeLayers(encoder, redundant_layers);    
    tempLayer = [flattenLayer("Name","encoder_out")];
    encoder_temp = addLayers(encoder,tempLayer);
    name_to_add = encoder.Layers(end, 1).Name;
    encoder = connectLayers(encoder_temp,name_to_add,"encoder_out");
    encoder = dlnetwork(encoder, Initialize=false);
    encoder = initialize(encoder);


    
end 




function [loss, gradientsE, gradientsD] = mseLoss(netE, netD, X)

    % Forward data through the dlnetwork object.
    encoder_out= forward(netE,X);

    decoder_out = forward(netD, encoder_out);
    % Compute loss.
    loss = mse(decoder_out, X);
    % Compute gradients.
    [gradientsE,gradientsD] = dlgradient(loss,netE.Learnables,netD.Learnables);

end
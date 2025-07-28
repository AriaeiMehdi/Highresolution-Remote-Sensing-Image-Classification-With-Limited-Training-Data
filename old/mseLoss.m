function [loss, gradients] = mseLoss(net, X)

    % Forward data through the dlnetwork object.
    Y = forward(net,X);

    % Compute loss.
    loss = mse(Y,X);

    % Compute gradients.
    gradients = dlgradient(loss,net.Learnables);

end


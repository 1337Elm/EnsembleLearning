import torch
import numpy as np


# Cumulative test loss is criteria for returning,
# if not fulfilled ever retrain until it is fulfilled
def training_nn(n_epochs, batchsize, model, optimizer, loss_fn,
                InputData, OutputData, input_test, output_test, tol):
    """Training a neural network, if cumulative test loss is
        under a given tolerance return the model otherwise return 0

    Args:
        n_epochs (int): max iterations
        batchsize (int): batchsize
        model (object): Neural Network
        optimizer (object): Optimizer function
        loss_fn (object): _description_
        InputData (torch tensor): data to train on
        OutputData (torch tensor): output data
        input_test (torch tensor): data to test on
        output_test (torch tensor): output data to test on
        tol (float): _description_

    Returns:
        model object: Trained Neural network
        0 int: If model not good enough return 0
    """
    loss_array = [1000]
    for epoch in range(n_epochs):
        # After 500 epochs update learning rate to 0.001
        if epoch == 5:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        model.train()
        for i in range(0, len(InputData), batchsize):
            X_batch = InputData[i:i+batchsize]
            y_pred = model(X_batch)

            loss = loss_fn[0](y_pred.squeeze(), torch.flatten(OutputData[i:
                                                              i + batchsize]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_val = []
        model.eval()
        for i in range(0, len(input_test), batchsize):
            X_batch = input_test[i:i + batchsize]
            y_pred = model(X_batch)

            # Wrong size some of the time??
            loss_val.append(loss_fn[0](y_pred.squeeze(),
                                       torch.flatten(output_test[i:
                                                     i+batchsize])
                                       .squeeze()).item())

        if epoch % 10 == 0:
            print(f"NN Epoch {epoch}: loss: {np.average(loss_val)}")

        if np.average(loss_val) <= tol:
            print(f"Successfull model loss: {np.average(loss_val)}")
            return model, loss_array[1:]

        if epoch > 50:
            if np.average(loss_val) > loss_array[-5]:
                print(f"NN Validation error rising! {np.average(loss_val)}")
                return model, loss_array[1:]

        loss_array.append(np.average(loss_val))
    return model, loss_array[1:]


def training_cnn(n_epochs, batchsize, model, optimizer, loss_fn,
                 InputData, OutputData, tol, input_test,
                 output_test, model_name):
    """Training a CNN

    Args:
        n_epochs (int): max number of iterations
        batchsize (int): ammount of data points to consider in each iteration
        model (object): the network
        optimizer (object): the optimizer function that updates
          the parameters in the network
        loss_fn (object): the loss function that calculates the loss,
          e.g mean square loss
        InputData (list): list of images to be used to train the network
        OutputData (list): vector of accurate C_d values to be used
          to train the network
        tol (float): stopping criteria for the model, if loss <= tol the model
          is deemed successfull and do not need further training
        input_test (list): list of images used to validate the model
        output_test (list): vector of accurate C_d values to be used
          to validate the model
        model_name (str): name of the model, used for printing more clearly

    Returns:
        model object: the trained network
        loss_array list: list of average loss values for each epoch used
         for convergence plots
    """
    loss_array = [1000]

    for epoch in range(n_epochs):
        if epoch == 3:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        model.train()
        for i in range(0, len(InputData), batchsize):
            X_batch = InputData[i:i + batchsize]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred.squeeze(), torch.flatten(OutputData[i:
                                                           i+batchsize]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_val = []
        model.eval()
        for i in range(0, len(input_test), batchsize):
            X_batch = input_test[i:i + batchsize]
            y_pred = model(X_batch)
            loss_val.append(loss_fn(y_pred.squeeze(),
                                    torch.flatten(output_test[i:i+batchsize])
                                    .squeeze()).item())

        if epoch % 10 == 0:
            print(f"CNN{model_name} Epoch {epoch}: loss: \
                  {np.average(loss_val)}")

        if np.average(loss_val) <= tol:
            print(f"Successfull model{model_name} loss: \
                  {np.average(loss_val)}")
            return model, loss_array[1:]

        if epoch > 5:
            if np.average(loss_val) > loss_array[-5]:
                print(f"CNN{model_name} Validation error rising! \
                       {np.average(loss_val)}")
                return model, loss_array[1:]

        loss_array.append(np.average(loss_val))

    return model, loss_array[1:]


def training_mmnn(n_epochs, batchsize, nn1, nn2, cnn1, cnn2, cnn3,
                  optimizer, loss_fn, InputData, OutputData,
                  input_test, output_test, tol):
    """Trains the multimodal neural network, using 1 feed-forward
     neural network and 3 convolutional neural networks
     that have already been trained

    Args:
        n_epochs (int): max number of iterations
        batchsize (int): ammount of data points to consider in each iteration
        nn1 (object): first neural network
        nn2 (object): second and final neural network
        cnn1 (object): first convolutional neural network
        cnn2 (object): second convolutional neural network
        cnn3 (object): third convolutional neural network
        optimizer (object): optimizer function that updates parameters
         in the network being trained
        loss_fn (object): calculates the loss for each epoch for the network
          being trained, e.g mean squared loss
        InputData (list): list of lists with data for each network to be used
         in order to train the final network
        OutputData (list): lists of vectors with accurate data used to train
          the final network
        input_test (list): lists of lists with data for each network to be used
          in order to validate the final network
        output_test (list): lists of vectors with accurate data
          used to validate the final network
        tol (float): stopping criteria for the model, if loss <= tol the model
          is deemed successfull and do not need further training

    Returns:
        models list: list of the networks used, trained
        loss_array list: list of average loss values for each epoch used for
          convergence plots
    """
    loss_array = [1000]

    y_pred1_train = nn1(InputData[0]).detach().numpy().squeeze()
    y_pred2_train = cnn1(InputData[1]).detach().numpy()[:, 0]
    y_pred3_train = cnn2(InputData[2]).detach().numpy()[:, 0]
    y_pred4_train = cnn3(InputData[3]).detach().numpy()[:, 0]

    y_pred1 = nn1(input_test[0]).detach().numpy().squeeze()
    y_pred2 = cnn1(input_test[1]).detach().numpy()[:, 0]
    y_pred3 = cnn2(input_test[2]).detach().numpy()[:, 0]
    y_pred4 = cnn3(input_test[3]).detach().numpy()[:, 0]

    preds = np.transpose(torch.tensor(np.array([y_pred1_train, y_pred2_train,
                                                y_pred3_train, y_pred4_train]),
                                      dtype=torch.float32))
    preds_validation = np.transpose(torch.tensor(np.array([y_pred1, y_pred2,
                                                           y_pred3, y_pred4]),
                                                 dtype=torch.float32))

    for epoch in range(n_epochs):
        if epoch == 5:
            optimizer = torch.optim.Adam(nn2.parameters(), lr=0.001)

        nn1.eval()
        cnn1.eval()
        cnn2.eval()
        cnn3.eval()

        nn2.train()
        for i in range(0, len(InputData[0]), batchsize):
            y_pred = nn2(preds[i:i + batchsize])
            loss = loss_fn(y_pred.squeeze(), torch.flatten(OutputData[i:
                                                           i + batchsize]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_val = []
        nn2.eval()
        for i in range(0, len(input_test[0]), batchsize):
            y_pred = nn2(preds_validation[i:i + batchsize])

            loss_val.append(loss_fn(y_pred.squeeze(),
                                    torch.flatten(output_test[i:i + batchsize])
                                    .squeeze()).item())

        if epoch % 10 == 0:
            print(f"MNN Epoch {epoch}: loss: {np.average(loss_val)}")

        if np.average(loss_val) <= tol:
            print(f"Successfull model loss: {np.average(loss_val)}")
            return [nn1, cnn1, cnn2, cnn3, nn2], loss_array[1:]

        if np.average(loss_val) > loss_array[-1]:
            print(f"MNN Validation error rising! {np.average(loss_val)}")
            return [nn1, cnn1, cnn2, cnn3, nn2], loss_array[1:]

        loss_array.append(np.average(loss_val))
    return [nn1, cnn1, cnn2, cnn3, nn2], loss_array[1:]

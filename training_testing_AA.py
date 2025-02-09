import torch
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


# Cumulative test loss is criteria for returning,
# if not fulfilled ever retrain until it is fulfilled
def training_nn(n_epochs, batchsize, model, optimizer,
                loss_fn, InputData, OutputData,
                input_test, output_test, tol):
    """Training a neural network, if cumulative test loss is under a given tolerance return the model, otherwise return 0

    Args:
        n_epochs (int): max iterations
        batchsize (int): batchsize
        model (object): Neural Network
        optimizer (object): Optimizer function
        loss_fn (object): Loss function to be used
        InputData (torch tensor): data to train on
        OutputData (torch tensor): output data
        input_test (torch tensor): data to test on
        output_test (torch tensor): output data to test on
        tol (float): tolerance

    Returns:
        model object: Trained Neural network
        0 int: If model not good enough return 0
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=20)

    loss_array = [1000]
    loss_array_train = []
    for epoch in range(n_epochs):
        # After 500 epochs update learning rate to 0.001
        if epoch == 5:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss_epoch = 0
        model.train()
        for i in range(0, len(InputData), batchsize):
            X_batch = InputData[i:i+batchsize]
            y_pred = model(X_batch)

            loss = loss_fn[0](y_pred.squeeze(),
                              torch.flatten(OutputData[i:
                                                       i + batchsize]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
        loss_array_train.append(loss_epoch / (len(InputData)/batchsize))

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

        if epoch % 20 == 0:
            #print(f"NN Epoch {epoch}: loss: {np.average(loss_val)}")
            pass
        if np.average(loss_val) <= tol:
            #print(f"Successfull model loss: {np.average(loss_val)}")
            return model, np.average(loss_val), \
                loss_array[1:], loss_array_train

        if epoch > 50:
            if np.average(loss_val) > loss_array[-5]:
                #print(f"NN Validation error rising! {np.average(loss_val)}")
                return model, np.average(loss_val), \
                    loss_array[1:], loss_array_train

        loss_array.append(np.average(loss_val))
    return model, np.average(loss_val), loss_array[1:], loss_array_train


def training_cnn(n_epochs, batchsize, model, optimizer,
                 loss_fn, InputData, OutputData, tol,
                 input_test, output_test, model_name):
    """Training a CNN

    Args:
        n_epochs (int): max number of iterations
        batchsize (int): ammount of data points to consider in each iteration
        model (object): the network
        optimizer (object): the optimizer function that updates the parameters in the network
        loss_fn (object): the loss function that calculates the loss, e.g mean square loss
        InputData (list): list of images to be used to train the network
        OutputData (list): vector of accurate C_d values to be used to train the network
        tol (float): stopping criteria for the model, if loss <= tol the model is deemed successfull and do not need further training
        input_test (list): list of images used to validate the model
        output_test (list): vector of accurate C_d values to be used to validate the model
        model_name (str): name of the model, used for printing more clearly

    Returns:
        model object: the trained network
        loss_array list: list of average loss values
        for each epoch used for convergence plots
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=20)


    loss_array = [1000]
    loss_array_train = []

    for epoch in range(n_epochs):
        if epoch == 5:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss_epoch = 0
        model.train()
        for i in range(0, len(InputData), batchsize):
            X_batch = InputData[i:i + batchsize]
            y_pred = model(X_batch)

            loss = loss_fn(y_pred.squeeze(),
                           torch.flatten(OutputData[:, i:i+batchsize]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        loss_array_train.append(loss_epoch / (len(InputData) / batchsize))

        loss_val = []
        model.eval()
        for i in range(0, len(input_test), batchsize):
            X_batch = input_test[i:i + batchsize]
            y_pred = model(X_batch)
            loss_val.append(loss_fn(y_pred.squeeze(),
                                    torch.flatten(output_test[:,
                                                              i:i+batchsize])
                                    .squeeze()).item())

        if epoch % 20 == 0:
            #print(f"{model_name} Epoch {epoch} | loss: {np.average(loss_val)}")
            pass
        if np.average(loss_val) <= tol:
            #print(f"Successfull model{model_name} loss: {np.average(loss_val)}")
            return model, np.average(loss_val), \
                loss_array[1:], loss_array_train

        if epoch > 10:
            if np.average(loss_val) > loss_array[-5]:
                #print(f"CNN{model_name} Validation error rising! {np.average(loss_val)}")
                return model, np.average(loss_val), \
                    loss_array[1:], loss_array_train

        loss_array.append(np.average(loss_val))

    return model, np.average(loss_val), \
        loss_array[1:], loss_array_train


def training_mmnn(n_epochs, batchsize, nn1, nn2, cnn1,
                  optimizer, loss_fn, InputData, OutputData,
                  input_test, output_test, tol):
    """Trains the multimodal neural network, using one feed-forward neural network and one convolutional neural networks that have already been trained

    Args:
        n_epochs (int): max number of iterations
        batchsize (int): ammount of data points to consider in each iteration
        nn1 (object): first neural network
        nn2 (object): second and final neural network
        cnn1 (object): first convolutional neural network
        optimizer (object): optimizer function that updates parameters
        in the network being trained
        loss_fn (object): calculates the loss for each epoch for the network being trained, e.g mean squared loss
        InputData (list): list of lists with data for each network to be used in order to train the final network
        OutputData (list): lists of vectors with accurate data used to train the final network
        input_test (list): lists of lists with data for each network to be used in order to validate the final network
        output_test (list): lists of vectors with accurate data used to validate the final network
        tol (float): stopping criteria for the model, if loss <= tol the model is deemed successfull and do not need further training

    Returns:
        models list: list of the networks used, trained
        loss_array list: list of average loss values
        for each epoch used for convergence plots
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nn1.to(device)
    nn2.to(device)
    cnn1.to(device)

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=20)

    loss_array = [1000]
    loss_array_train = []

    y_pred1_train = nn1(InputData[0]).cpu().detach().numpy().squeeze()
    y_pred2_train = cnn1(InputData[1]).cpu().detach().numpy()[:, 0]

    y_pred1 = nn1(input_test[0]).cpu().detach().numpy().squeeze()
    y_pred2 = cnn1(input_test[1]).cpu().detach().numpy()[:, 0]

    preds = torch.tensor(np.transpose(np.array([y_pred1_train,
                                                y_pred2_train])),
                         dtype=torch.float32).to(device)
    preds_validation = torch.tensor(np.transpose(np.array([y_pred1,
                                                           y_pred2])),
                                    dtype=torch.float32).to(device)

    for epoch in range(n_epochs):
        if epoch == 5:
            optimizer = torch.optim.Adam(nn2.parameters(), lr=0.001)

        loss_epoch = 0
        nn1.eval()
        cnn1.eval()

        nn2.train()
        for i in range(0, len(InputData[0]), batchsize):
            y_pred = nn2(preds[i:i + batchsize])
            loss = loss_fn(y_pred.squeeze(),
                           torch.flatten(OutputData[i:i + batchsize]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        loss_array_train.append(loss_epoch / (len(InputData)/batchsize))

        loss_val = []
        nn2.eval()
        for i in range(0, len(input_test[0]), batchsize):
            y_pred = nn2(preds_validation[i:i + batchsize])

            loss_val.append(loss_fn(y_pred.squeeze(),
                                    torch.flatten(output_test[i:i + batchsize])
                                    .squeeze()).item())

        if epoch % 10 == 0:
            #print(f"MNN Epoch {epoch}: loss: {np.average(loss_val)}")
            pass
        if np.average(loss_val) <= tol:
            #print(f"Successfull model loss: {np.average(loss_val)}")
            torch.save(nn1.state_dict(), "./models/nn1.pth")
            torch.save(nn2.state_dict(), "./models/nn2.pth")
            torch.save(cnn1.state_dict(), "./models/cnn1.pth")
            return [nn1, cnn1, nn2], loss_array[1:], loss_array_train

        if epoch >= 5:
            if np.average(loss_val) > loss_array[-5]:
                #print(f"MNN Validation error rising! {np.average(loss_val)}")
                torch.save(nn1.state_dict(), "./models/nn1.pth")
                torch.save(nn2.state_dict(), "./models/nn2.pth")
                torch.save(cnn1.state_dict(), "./models/cnn1.pth")
                return [nn1, cnn1, nn2], loss_array[1:], loss_array_train

        loss_array.append(np.average(loss_val))
    torch.save(nn1.state_dict(), "./models/nn1.pth")
    torch.save(nn2.state_dict(), "./models/nn2.pth")
    torch.save(cnn1.state_dict(), "./models/cnn1.pth")
    return [nn1, cnn1, nn2], loss_array[1:], loss_array_train

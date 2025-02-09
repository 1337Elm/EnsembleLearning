import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import NeuralNetwork, ConvolutionalNeuralNetwork
from utils import read_csv, split_data, import_pictures
from training_testing_AA import training_cnn, training_nn, training_mmnn
from plotting_AA import showcase_model, convergence_plot, statistics
import matplotlib.pyplot as plt


def main():
    """Define parameters and control the process of creating the model(s)
    """
    # Initial data, to be determined
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    folder_nameXY = "./Pictures/2024-04-03/xy/"
    folder_nameXZ = "./Pictures/2024-04-03/xz/"
    folder_nameYZ = "./Pictures/2024-04-03/yz/"
    filename_data = "data2024-04-03.csv"

    # Parameters nn
    input_size, hidden_size, output_size = 3, 10, 1

    # Parameters cnn
    image_size, channels = 64, 3

    # Parameters final nn
    input_sizeFinal, hidden_sizeFinal, output_sizeFinal = 2, 8, 1

    # General training parameters
    batchsize, batchsize_cnn, n_epochs, tol = 32, 16, 1000, 1e-3

    # Test/Training split
    split, validation_split = 0.8, 0.9

    # Init the networks
    nn1 = NeuralNetwork(input_size, hidden_size, output_size)

    nnFinal = NeuralNetwork(input_sizeFinal,
                            hidden_sizeFinal, output_sizeFinal)

    # Loss functions for the networks (?)
    loss_fn = [nn.MSELoss(), nn.MSELoss()]

    # Optimizers for the networks
    optimizer = [optim.Adam(nn1.parameters(), lr=0.001),
                 optim.Adam(nnFinal.parameters(), lr=0.001)]

    # Parse data from file
    matrixInput, C_d_vector = read_csv(filename_data)

    # matrixInput = scale_data(matrixInput)

    X_train, X_test, Y_train, Y_test = split_data(matrixInput.transpose(),
                                                  C_d_vector, split)
    X_train_train, X_validation, Y_train_train, Y_validation = split_data(
                                        X_train, Y_train, validation_split)

    X_train_train_tensor = torch.tensor(X_train_train,
                                        dtype=torch.float32).to(device)
    X_validation_tensor = torch.tensor(X_validation,
                                       dtype=torch.float32).to(device)

    Y_train_train_tensor = torch.tensor(Y_train_train,
                                        dtype=torch.float32).to(device)
    Y_validation_tensor = torch.tensor(Y_validation,
                                       dtype=torch.float32).to(device)

    X_test_tensor = torch.tensor(X_test,
                                 dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test,
                                 dtype=torch.float32).to(device)

    nn_model, model_loss, \
        loss_array, loss_array_train = training_nn(n_epochs,
                                                   batchsize, nn1,
                                                   optimizer[0], loss_fn,
                                                   X_train_train_tensor,
                                                   Y_train_train_tensor,
                                                   X_validation_tensor,
                                                   Y_validation_tensor, tol)

    # Train CNN with the error from NN
    diff_train = torch.tensor(np.array([Y_train_train -
                                        nn_model(X_train_train_tensor)
                                        .cpu().detach().numpy().squeeze()]),
                              dtype=torch.float32).to(device)
    diff_validation = torch.tensor(np.array([Y_validation -
                                             nn_model(X_validation_tensor)
                                             .cpu().detach()
                                             .numpy().squeeze()]),
                                   dtype=torch.float32).to(device)

    cnn_AA = ConvolutionalNeuralNetwork(image_size, channels)

    loss_fn_cnn_AA = nn.MSELoss()
    optimizer_cnn_AA = optim.Adam(cnn_AA.parameters(), lr=0.001)

    data_AA = np.array(import_pictures([folder_nameXY,
                                        folder_nameXZ, folder_nameYZ]))

    data_tensor_AA = torch.tensor(data_AA, dtype=torch.float32).to(device)

    X_train_AA, X_test_AA, \
        Y_train_AA, Y_test_AA = split_data(data_tensor_AA,
                                           torch.tensor(C_d_vector,
                                                        dtype=torch.float32),
                                           split)

    X_train_train_AA, X_validation_AA, \
        Y_train_train_AA, Y_validation_AA = split_data(X_train_AA,
                                                       Y_train_AA,
                                                       validation_split)

    cnn_model_AA, cnn_mod_AA_loss, \
        cnn_mod_loss_array, \
        cnn_loss_array_train_AA = training_cnn(n_epochs,
                                               batchsize_cnn,
                                               cnn_AA, optimizer_cnn_AA,
                                               loss_fn_cnn_AA,
                                               X_train_train_AA,
                                               diff_train,
                                               tol, X_validation_AA,
                                               diff_validation, "CNNAA")

    # Train meta network
    mnn, mnn_loss_array, \
        mnn_loss_array_train = training_mmnn(n_epochs, batchsize, nn_model,
                                             nnFinal, cnn_model_AA,
                                             optimizer[-1], loss_fn[-1],
                                             [X_train_train_tensor,
                                              X_train_train_AA],
                                             Y_train_train_tensor,
                                             [X_validation_tensor,
                                              X_validation_AA],
                                             Y_validation_tensor, tol)

    # Print statistical metrics for all networks
    statistics(nn_model, X_test_tensor, Y_test_tensor, "NN")
    statistics(cnn_model_AA, X_test_AA, Y_test_AA, "CNN")
    statistics(mnn, [X_test_tensor, X_test_AA], Y_test_tensor, "MNN")

    # Create plots for network models
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 6), sharey=False)
    fig1.tight_layout()
    plt.subplots_adjust(top=0.90)

    diff_test = torch.tensor(np.array([Y_test -
                                       nn_model(X_test_tensor).cpu().detach()
                                       .numpy().squeeze()]),
                             dtype=torch.float32)

    showcase_model(nn_model, X_test_tensor, Y_test_tensor,
                   "DNN", 0, [ax1, ax2, ax3])
    showcase_model(cnn_model_AA, [X_test_AA, X_test_tensor],
                   diff_test, "CNN", 1, [ax1, ax2, ax3])
    showcase_model(mnn, [X_test_tensor, X_test_AA],
                   Y_test_tensor, "MNN", 2, [ax1, ax2, ax3])

    plt.suptitle("Predictions")
    plt.savefig("./models/plots/prediction_NN_CNNAA_MNN_error_train.png",
                bbox_inches='tight')

    convergence_plot(loss_array, cnn_mod_loss_array, mnn_loss_array)

    plt.figure()

    predictions = nn_model(X_test_tensor).cpu().detach().numpy()
    Re = X_test_tensor.cpu().numpy().transpose()[2]

    plt.scatter(Re, predictions, marker='^', label="$C_{d-Pred}$", s=10)
    plt.scatter(Re, Y_test_tensor.cpu(), marker='o', facecolors='none',
                edgecolors='k', label="$C_d$", s=10)
    plt.ylabel("$C_d$")
    plt.xlabel("Re")
    plt.legend()
    plt.title("DNN Predictions")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("./models/plots/prediction_NN_only.png")

    epochs = np.linspace(0, len(loss_array), len(loss_array))
    plt.figure()

    plt.plot(epochs, loss_array, 'b-.')
    plt.plot(epochs[::5], loss_array[::5], 'b^', linewidth=0.01)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Deep Neural Network")
    plt.yscale("log")
    plt.savefig("./models/plots/convergence_NN.png")

    epochs_mnn = np.linspace(0, len(mnn_loss_array), len(mnn_loss_array))
    epochs_cnn = np.linspace(0, len(cnn_mod_loss_array),
                             len(cnn_mod_loss_array))

    plt.figure()

    plt.plot(epochs, loss_array, 'b-.', label="DNN")
    plt.plot(epochs[::5], loss_array[::5], 'b^', linewidth=0.01)

    plt.plot(epochs_mnn, mnn_loss_array, 'k-.', label="Stacked Network")
    plt.plot(epochs_mnn[::5], mnn_loss_array[::5], 'k^', linewidth=0.01)

    plt.plot(epochs_cnn, cnn_mod_loss_array, 'r-.', label="CNN")
    plt.plot(epochs_cnn[::5], cnn_mod_loss_array[::5], 'r^', linewidth=0.01)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Convergence")
    plt.yscale("log")
    plt.legend()
    plt.savefig("./models/plots/convergence_MNN.png")


if __name__ == '__main__':
    main()

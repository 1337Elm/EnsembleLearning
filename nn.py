import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import multiprocessing
from networks import NeuralNetwork, ConvolutionalNeuralNetwork
from data_handling import read_csv, split_data, import_pictures
from training_testing import training_cnn, training_nn, training_mmnn
from plotting import showcase_model, convergence_plot, statistics
import matplotlib.pyplot as plt


def main():
    """Inits networks, trains them and finally
      plots results as well as statistics
    """
    # Initial data, to be determined

    folder_nameXY = "./Pictures/2024-04-11/xy/"
    folder_nameXZ = "./Pictures/2024-04-11/xz/"
    folder_nameYZ = "./Pictures/2024-04-11/yz/"
    filename_data = "data2024-04-11.csv"

    # Parameters nn
    input_size = 4
    hidden_size = 10
    output_size = 1

    # Parameters cnn
    image_size = 64
    channels = 1

    # Parameters final nn
    input_sizeFinal = 4  # number of layer 0 network models
    hidden_sizeFinal = 8
    output_sizeFinal = 1

    # General training parameters
    batchsize = 32  # To be determined
    batchsize_cnn = 16  # i.e 32 images
    n_epochs = 1000
    tol = 1e-3

    # Test/Training split
    split = 0.8
    validation_split = 0.9

    # Init the networks
    nn1 = NeuralNetwork(input_size, hidden_size, output_size)
    cnn1 = ConvolutionalNeuralNetwork(image_size, channels)
    cnn2 = ConvolutionalNeuralNetwork(image_size, channels)
    cnn3 = ConvolutionalNeuralNetwork(image_size, channels)

    nnFinal = NeuralNetwork(input_sizeFinal, hidden_sizeFinal,
                            output_sizeFinal)

    # Loss functions for the networks (?)
    loss_fn = [nn.MSELoss(), nn.MSELoss(),
               nn.MSELoss(), nn.MSELoss(), nn.MSELoss()]

    # Optimizers for the networks
    optimizer = [optim.Adam(nn1.parameters(), lr=0.001),
                 optim.Adam(cnn1.parameters(), lr=0.0001),
                 optim.Adam(cnn2.parameters(), lr=0.0001),
                 optim.Adam(cnn3.parameters(), lr=0.0001),
                 optim.Adam(nnFinal.parameters(), lr=0.001)]

    # Parse data from file
    matrixInput, matrixOutput, C_d_vector = read_csv(filename_data)

    X_train, X_test, Y_train, Y_test = split_data(matrixInput.transpose(),
                                                  C_d_vector, split)
    X_train_train, X_validation, \
        Y_train_train, Y_validation = split_data(X_train,
                                                 Y_train, validation_split)

    X_train_train_tensor = torch.tensor(X_train_train, dtype=torch.float32)
    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)

    Y_train_train_tensor = torch.tensor(Y_train_train, dtype=torch.float32)
    Y_validation_tensor = torch.tensor(Y_validation, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    nn_model, loss_array = training_nn(n_epochs, batchsize, nn1,
                                       optimizer[0], loss_fn,
                                       X_train_train_tensor,
                                       Y_train_train_tensor,
                                       X_validation_tensor,
                                       Y_validation_tensor, tol)

    # Train CNN with the error from NN
    # diff_train = torch.tensor(np.array([Y_train_train -
    #  nn_model(X_train_train_tensor).detach().numpy().squeeze()]),
    #  dtype = torch.float32)
    # diff_validation = torch.tensor(np.array([Y_validation -
    #  nn_model(X_validation_tensor).detach().numpy().squeeze()]),
    # dtype = torch.float32)

    # LOAD PICS XY
    data = np.array(import_pictures(folder_nameXY))
    data_tensor = torch.tensor(data, dtype=torch.float32)

    X_train_pic, X_test_pic, \
        Y_train_pic, Y_test_pic = split_data(data_tensor,
                                             torch.tensor(C_d_vector,
                                                          dtype=torch.float32),
                                             split)
    X_train_train_pic, X_validation_pic, \
        Y_train_train_pic, Y_validation_pic = split_data(X_train_pic,
                                                         Y_train_pic,
                                                         validation_split)

    # LOAD PICS XZ
    dataXZ = np.array(import_pictures(folder_nameXZ))
    data_tensorXZ = torch.tensor(dataXZ, dtype=torch.float32)

    X_train_picXZ, \
        X_test_picXZ, \
        Y_train_picXZ, \
        Y_test_picXZ = split_data(data_tensorXZ,
                                  torch.tensor(C_d_vector,
                                               dtype=torch.float32),
                                  split)
    X_train_train_picXZ, X_validation_picXZ, \
        Y_train_train_picXZ, Y_validation_picXZ = split_data(X_train_picXZ,
                                                             Y_train_picXZ,
                                                             validation_split)

    # LOAD PICS YZ
    dataYZ = np.array(import_pictures(folder_nameYZ))
    data_tensorYZ = torch.tensor(dataYZ, dtype=torch.float32)

    X_train_picYZ, X_test_picYZ, \
        Y_train_picYZ, \
        Y_test_picYZ = split_data(data_tensorYZ,
                                  torch.tensor(C_d_vector,
                                               dtype=torch.float32),
                                  split)
    X_train_train_picYZ, X_validation_picYZ, \
        Y_train_train_picYZ, \
        Y_validation_picYZ = split_data(X_train_picYZ,
                                        Y_train_picYZ, validation_split)

    # Easily scalabe for more models
    q = multiprocessing.Queue()
    cnn_model1 = multiprocessing.Process(target=run_func,
                                         args=(training_cnn, q,
                                               (n_epochs, batchsize_cnn,
                                                cnn1, optimizer[1], loss_fn[1],
                                                X_train_train_pic,
                                                Y_train_train_pic,
                                                tol, X_validation_pic,
                                                Y_validation_pic, 1), 1))
    cnn_model2 = multiprocessing.Process(target=run_func,
                                         args=(training_cnn, q,
                                               (n_epochs, batchsize_cnn,
                                                cnn2, optimizer[1], loss_fn[1],
                                                X_train_train_picXZ,
                                                Y_train_train_picXZ, tol,
                                                X_validation_picXZ,
                                                Y_validation_picXZ, 2), 2))
    cnn_model3 = multiprocessing.Process(target=run_func,
                                         args=(training_cnn, q,
                                               (n_epochs, batchsize_cnn,
                                                cnn3, optimizer[1], loss_fn[1],
                                                X_train_train_picYZ,
                                                Y_train_train_picYZ, tol,
                                                X_validation_picYZ,
                                                Y_validation_picYZ, 3), 3))

    # and and subprocces to list of processes
    processes = [cnn_model1, cnn_model2, cnn_model3]
    for p in processes:
        p.start()
    results = [q.get() for p in processes]
    for p in processes:
        p.join()

    for r in results:
        if r[0] == 1:
            cnn_model, cnn_loss_array = r[1]
        elif r[0] == 2:
            cnn_model2, cnn_loss_array2 = r[1]
        elif r[0] == 3:
            cnn_model3, cnn_loss_array3 = r[1]

    mnn, mnn_loss_array = training_mmnn(n_epochs, batchsize, nn_model, nnFinal,
                                        cnn_model, cnn_model2, cnn_model3,
                                        optimizer[-1], loss_fn[-1],
                                        [X_train_train_tensor,
                                         X_train_train_pic,
                                         X_train_train_picXZ,
                                         X_train_train_picYZ],
                                        Y_train_train_tensor,
                                        [X_validation_tensor,
                                         X_validation_pic,
                                         X_validation_picXZ,
                                         X_validation_picYZ],
                                        Y_validation_tensor, tol)

    statistics(nn_model, X_test_tensor, Y_test_tensor, "NN")
    statistics(cnn_model, X_test_pic, Y_test_pic, "CNN")
    statistics(mnn, [X_test_tensor, X_test_pic, X_test_picXZ, X_test_picYZ],
               Y_test_tensor, "MNN")

    fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5,
                                                   figsize=(12, 6),
                                                   sharey=True)
    fig1.tight_layout()
    plt.subplots_adjust(top=0.90)

    showcase_model(nn_model, X_test_tensor, Y_test_tensor, "NN", 0,
                   [ax1, ax2, ax3, ax4, ax5])
    showcase_model(cnn_model, [X_test_pic, X_test_tensor], Y_test_tensor,
                   "CNN", 1, [ax1, ax2, ax3, ax4, ax5])
    showcase_model(cnn_model2, [X_test_picXZ, X_test_tensor], Y_test_tensor,
                   "CNN", 2, [ax1, ax2, ax3, ax4, ax5])
    showcase_model(cnn_model3, [X_test_picYZ, X_test_tensor], Y_test_tensor,
                   "CNN", 3, [ax1, ax2, ax3, ax4, ax5])
    showcase_model(mnn, [X_test_tensor, X_test_pic, X_test_picXZ,
                         X_test_picYZ],
                   Y_test_tensor, "MNN", 4, [ax1, ax2, ax3, ax4, ax5])

    plt.suptitle("Predictions")
    plt.savefig("./models/plots/prediction.png", bbox_inches='tight')

    convergence_plot(loss_array, cnn_loss_array, cnn_loss_array2,
                     cnn_loss_array3, mnn_loss_array)

    plt.show()


# For parallel processing
def run_func(func, result_queue, args, process_id):
    id = process_id
    result_queue.put((id, func(*args)))


if __name__ == '__main__':
    main()

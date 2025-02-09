import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import multiprocessing
from networks import NeuralNetwork
from networks import ConvolutionalNeuralNetwork
from data_handling_AA import read_csv
from data_handling_AA import split_data
from data_handling_AA import import_pictures
from training_testing_AA import training_cnn
from training_testing_AA import training_nn
    """This files was used early in the project to create onxx files 
       which was used in netron to create visual structure of model

       It is not up to date and should not be used since much of the code is redundant and hardcoded
    """

# Input data

folder_nameXY = "./Pictures/2024-04-03/xy/"
folder_nameXZ = "./Pictures/2024-04-03/xz/"
folder_nameYZ = "./Pictures/2024-04-03/yz/"
filename_data = "data2024-04-03.csv"

# Parameters nn
input_size = 4 
hidden_size = 10
output_size = 1

# Parameters cnn
image_size = 64
channels = 3

# Parameters final nn
input_sizeFinal = 2  # number of layer 0 network models
hidden_sizeFinal = 8
output_sizeFinal = 1

# General training parameters
batchsize = 32  # To be determined
batchsize_cnn = 16  # i.e 32 images
n_epochs = 1
tol = 1e-3

# Test/Training split
split = 0.8
validation_split = 0.9

# Init the networks
nn1 = NeuralNetwork(input_size, hidden_size, output_size)

nnFinal = NeuralNetwork(input_sizeFinal, hidden_sizeFinal, output_sizeFinal)

# Loss functions for the networks (?)
loss_fn = [nn.MSELoss(),nn.MSELoss()]

# Optimizers for the networks
optimizer = [optim.Adam(nn1.parameters(), lr=0.001),optim.Adam(nnFinal.parameters(), lr=0.001)]

# Parse data from file
matrixInput, matrixOutput, C_d_vector = read_csv(filename_data)

X_train, X_test, Y_train, Y_test = split_data(matrixInput.transpose(), C_d_vector, split)
X_train_train, X_validation, Y_train_train, Y_validation= split_data(X_train, Y_train, validation_split)

X_train_train_tensor = torch.tensor(X_train_train, dtype=torch.float32)
X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)

Y_train_train_tensor = torch.tensor(Y_train_train, dtype=torch.float32)
Y_validation_tensor = torch.tensor(Y_validation, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

nn_model, model_loss, loss_array, loss_array_train = training_nn(n_epochs,batchsize, nn1, optimizer[0], loss_fn,
                        X_train_train_tensor, Y_train_train_tensor, X_validation_tensor, Y_validation_tensor, tol)

# Train CNN with the error from NN
diff_train = torch.tensor(np.array([Y_train_train - nn_model(X_train_train_tensor).detach().numpy().squeeze()]), dtype=torch.float32)
diff_validation = torch.tensor(np.array([Y_validation - nn_model(X_validation_tensor).detach().numpy().squeeze()]), dtype=torch.float32)

cnn_AA = ConvolutionalNeuralNetwork(image_size, channels)

loss_fn_cnn_AA = nn.MSELoss()
optimizer_cnn_AA = optim.Adam(cnn_AA.parameters(), lr=0.001)

data_AA = np.array(import_pictures([folder_nameXY, folder_nameXZ, folder_nameYZ]))
data_tensor_AA = torch.tensor(data_AA, dtype=torch.float32)

X_train_AA, X_test_AA, Y_train_AA, Y_test_AA = split_data(data_tensor_AA,torch.tensor(C_d_vector, dtype=torch.float32), split)

X_train_train_AA, X_validation_AA, Y_train_train_AA, Y_validation_AA = split_data(X_train_AA, Y_train_AA, validation_split)

cnn_model_AA, cnn_mod_AA_loss, cnn_mod_loss_array, cnn_loss_array_train_AA = training_cnn(n_epochs,batchsize_cnn,
            cnn_AA, optimizer_cnn_AA, loss_fn_cnn_AA, X_train_train_AA, diff_train, tol, X_validation_AA, diff_validation,"CNNAA")

nn_model.eval()
cnn_model_AA.eval()

dummy_input = torch.randn(1, 4, requires_grad=True)
dummy_inputcnn = torch.randn(3, 3, 16777216, 1, requires_grad=True)

torch.onnx.export(nn_model,          # model being run
                  dummy_input,       # model input (or a tuple for multiple inputs)
                  "NN1.onnx",        # where to save the model
                  export_params=False,  # store the trained parameter weights inside the model file
                  opset_version=11,     # the ONNX version to export the model to
                  do_constant_folding=True,     # whether to execute constant folding for optimization
                  input_names=['modelInput'],   # the model's input names
                  output_names=['modelOutput'], # the model's output names
                  dynamic_axes={'modelInput': {0: '32'},    # variable length axes
                                'modelOutput': {0: '32'}})

torch.onnx.export(cnn_model_AA,  # model being run
         X_validation_AA,        # model input (or a tuple for multiple inputs)
         "CNN_AA_1.onnx",        # where to save the model
         export_params=False,    # store the trained parameter weights inside the model file
         opset_version=10,       # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],    # the model's input names
         output_names = ['modelOutput'],  # the model's output names
         dynamic_axes={'modelInput': {0: '32'},    # variable length axes
                       'modelOutput': {0: '32'}})

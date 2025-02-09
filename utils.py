import pyvista as pv
import numpy as np
import pandas as pd
import os
import imageio
from sklearn.model_selection import train_test_split
from random import randrange
from sklearn.preprocessing import StandardScaler
from networks import NeuralNetwork, ConvolutionalNeuralNetwork
import torch
from training_testing_AA import training_nn, training_cnn, training_mmnn
import sys


class EnsembleModel():
    """Class that simplifies the prediction process
        for the ensemble network model. Simply load previously
        trained models and predict values or train them and then predict.

        :param root: path to root folder
        :type root: str
        :param device: device to which model and data is loaded
        :type device: torch.device

        :param loaded: if the ensembled model has loaded weights and is ready for making prediction

        :type loaded: bool
    """

    def __init__(self, nn_dim=[3, 10, 1], cnn_dim=[64, 3],
                 mnn_dim=[2, 8, 1], root: str = os.getcwd(),
                 device: torch.device = "cpu"):
        """Constructor method

        Args:
            nn_dim (list, optional): Dimensions for the first neural network, input layer, hidden layer and output layer. Defaults to [3, 10, 1].
            cnn_dim (list, optional): Dimensions for the convolutional neural network, input layer, hidden layer and output layer. Defaults to [64, 3].
            mnn_dim (list, optional): Dimensions for the final neural network, input layer, hidden layer and output layer. Defaults to [2, 8, 1].
            root (str, optional): Path to current working directory. Defaults to os.getcwd().
            device (torch.device, optional): pecify if CPU or GPU should be used. Defaults to "cpu".
        """
        self.root = root
        self.device = device

        self.nn_dim = nn_dim
        self.cnn_dim = cnn_dim
        self.mnn_dim = mnn_dim

        self.nn1 = NeuralNetwork(nn_dim[0],
                                 nn_dim[1],
                                 nn_dim[2]).to(device)
        self.cnn = ConvolutionalNeuralNetwork(cnn_dim[0],
                                              cnn_dim[1]).to(device)
        self.nn2 = NeuralNetwork(mnn_dim[0],
                                 mnn_dim[1],
                                 mnn_dim[2]).to(device)

        self.loaded = False

    def load_model(self, nn1_path: str = "/models/nn1.pth",
                   nn2_path: str = "/models/nn2.pth",
                   cnn_path: str = "/models/cnn1.pth"):
        """Loads wights from saved model

        Args:
            nn1_path (str, optional): path to nn1 state dict. Defaults to "/models/nn1.pth".

            nn2_path (str, optional): path to nn2 state dict. Defaults to "/models/nn2.pth".

            cnn_path (str, optional): path to cnn state dict. Defaults to "/models/cnn1.pth".
        """
        PATH_nn1 = self.root + nn1_path
        PATH_cnn = self.root + cnn_path
        PATH_nn2 = self.root + nn2_path

        assert os.path.exists(os.path.join(self.root, PATH_nn1)), "nn1 path\
            does not exist"
        assert os.path.exists(os.path.join(self.root, PATH_cnn)), "cnn path\
            does not exist"
        assert os.path.exists(os.path.join(self.root, PATH_nn2)), "nn2 path\
            does not exist"

        nn1 = NeuralNetwork(self.nn_dim[0],
                            self.nn_dim[1],
                            self.nn_dim[2]).to(self.device)
        cnn = ConvolutionalNeuralNetwork(self.cnn_dim[0],
                                         self.cnn_dim[1]).to(self.device)
        nn2 = NeuralNetwork(self.mnn_dim[0],
                            self.mnn_dim[1],
                            self.mnn_dim[2]).to(self.device)

        try:
            nn1.load_state_dict(
                torch.load(PATH_nn1,
                           weights_only=True,
                           map_location=torch.device(self.device)))
            cnn.load_state_dict(
                torch.load(PATH_cnn,
                           weights_only=True,
                           map_location=torch.device(self.device)))
            nn2.load_state_dict(
                torch.load(PATH_nn2,
                           weights_only=True,
                           map_location=torch.device(self.device)))
        except Exception as e:
            print(e)
            print("Model dimensions changed, retrain networks \
                  or use pre-trained networks")
            sys.exit(1)

        nn1.eval()
        cnn.eval()
        nn2.eval()

        self.nn1 = nn1
        self.cnn = cnn
        self.nn2 = nn2

        self.loaded = True

    def train(self, filename="data2024-04-03.csv",
              split=0.8, validation_split=0.9,
              n_epochs=1000, batchsize=32,
              batchsize_cnn=16, tol=1e-3,
              optimizer=torch.optim.Adam, loss_fn=torch.nn.MSELoss,
              optimizer_cnn=torch.optim.Adam, loss_fn_cnn=torch.nn.MSELoss,
              lr=0.001, folder_nameXY="./Pictures/2024-04-03/xy/",
              folder_nameXZ="./Pictures/2024-04-03/xz/",
              folder_nameYZ="./Pictures/2024-04-03/yz/"):
        """Trains the ensemble model, using standard values unless otherwise specified.

        Args:
            filename (str, optional):Filename of the tabular data (csv). Defaults to "data2024-04-03.csv".
            split (float, optional): ratio of train/test set size. Defaults to 0.8.
            validation_split (float, optional): ratio of train/val size. Defaults to 0.9.
            n_epochs (int, optional): Maximum number of epochs to train for. Defaults to 1000.
            batchsize (int, optional): Number of epochs to train model. Defaults to 32.
            batchsize_cnn (int, optional): Batchsize of neural network. Defaults to 16.
            tol (_type_, optional): Error tolerance before training is stopped. Defaults to 1e-3.
            optimizer (_type_, optional): Torch optimizer for training (for nn). Defaults to torch.optim.Adam.
            loss_fn (_type_, optional): Loss function for training (for nn). Defaults to torch.nn.MSELoss.
            optimizer_cnn (_type_, optional): Torch optimizer for training (for cnn). Defaults to torch.optim.Adam.
            loss_fn_cnn (_type_, optional): Loss function for training (for cnn). Defaults to torch.nn.MSELoss.
            lr (float, optional): learning rate. Defaults to 0.001.
            folder_nameXY (str, optional): Folder path to images in XY direction. Defaults to "./Pictures/2024-04-03/xy/".
            folder_nameXZ (str, optional): Folder path to images in XZ direction. Defaults to "./Pictures/2024-04-03/xz/".
            folder_nameYZ (str, optional): Folder path to images in YZ direction. Defaults to "./Pictures/2024-04-03/yz/".
        """

        # Fix data for neural network
        matrixInput, C_d_vector = read_csv(filename)

        X_t, _, Y_t, _ = split_data(matrixInput.transpose(),
                                    C_d_vector, split)
        X_t_t, X_val, Y_t_t, Y_val = split_data(X_t, Y_t, validation_split)

        X_t_t_T = torch.tensor(X_t_t, dtype=torch.float32).to(self.device)
        X_val_T = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        Y_t_t_T = torch.tensor(X_t_t_T, dtype=torch.float32).to(self.device)
        Y_val_T = torch.tensor(Y_val, dtype=torch.float32).to(self.device)

        optimizer = [optimizer(self.nn1.parameters(), lr=lr),
                     optimizer(self.nn2.parameters(), lr=lr)]

        # Train first neural networks
        nn1, _, _ = training_nn(n_epochs,
                                batchsize, self.nn1,
                                optimizer[0], loss_fn,
                                X_t_t_T, Y_t_t_T,
                                X_val_T, Y_val_T, tol)

        diff_train = torch.tensor(np.array([Y_t_t -
                                            nn1(X_t_t_T)
                                            .cpu().detach()
                                            .numpy().squeeze()]),
                                  dtype=torch.float32).to(self.device)
        diff_validation = torch.tensor(np.array([Y_val -
                                                 nn1(X_val_T)
                                                 .cpu().detach()
                                                 .numpy().squeeze()]),
                                       dtype=torch.float32).to(self.device)

        data_AA = np.array(import_pictures([folder_nameXY,
                                            folder_nameXZ, folder_nameYZ]))

        data_tensor_AA = torch.tensor(data_AA,
                                      dtype=torch.float32).to(self.device)

        X_t_AA, _, Y_t_AA, _ = split_data(data_tensor_AA,
                                          torch.tensor(C_d_vector,
                                                       dtype=torch.float32),
                                          split)

        X_t_t_AA, X_val_AA, _, _ = split_data(X_t_AA, Y_t_AA, validation_split)

        optimizer_cnn = optimizer_cnn(self.cnn.parameters(), lr=0.001)

        # Train CNN
        cnn, _, _, _ = training_cnn(n_epochs,
                                    batchsize_cnn,
                                    self.cnn, optimizer_cnn,
                                    loss_fn_cnn,
                                    X_t_t_AA,
                                    diff_train,
                                    tol, X_val_AA,
                                    diff_validation, "CNNAA")

        # Train ensemble model
        mnn, _, _ = training_mmnn(n_epochs, batchsize, nn1,
                                  self.nn2, cnn,
                                  optimizer[-1], loss_fn[-1],
                                  [X_t_t_T, X_t_t_AA],
                                  Y_t_t_T,
                                  [X_val_T, X_val_AA],
                                  Y_val_T, tol)

        self.nn1 = nn1
        self.cnn = cnn
        self.nn2 = mnn[-1]

    def predict(self, input_nn, input_cnn) -> np.array:
        """Predicts values using the ensemble model

        Args:
            input_nn (torch.tensor): Torch tensor containing input data
            for the neural network
            input_cnn (torch.tensor): Torch tensor containing input data
            for the convolutional neural network, i.e. images

        Returns:
            np.array: np.array of predicted values for drag coefficient
        """
        if self.loaded:
            nn1_pred = self.nn1(input_nn).cpu().detach().numpy().squeeze()
            cnn_pred = self.cnn(input_cnn).cpu().detach().numpy().squeeze()

            predictions = torch.tensor(np.transpose(np.array([nn1_pred,
                                                    cnn_pred])),
                                       dtype=torch.float32).to(self.device)

            final_prediction = self.nn2(predictions).cpu().detach()\
                                                    .numpy().squeeze()
        else:
            print("Model not loaded, please load or train model to predict")

        return final_prediction


def picture_generator(P: int, C: int, i: int, polydata: pv.PolyData,
                      date: str, posXY: tuple = None,
                      posXZ: tuple = None, posYZ: tuple = None):
    """Function generating 3 pictures for each simulated particle from different angles.

    Args:
        P (int): number of sub-spheres in the particle, for filenamning
        C (int): Several configurations are created with the same number
          of sub-spheres, the indicates which configuration.
        i (int): Index of datapoint
        polydata (pv.PolyData): pyvista PolyData object,
          used to render the particle
        date (str): The date when the datageneration started,
          used for file naming
        posXY (tuple, optional): Camera position when taking screenshot in XY.
          Defaults to None.
        posXZ (tuple, optional): Camera position when taking screenshot in XZ.
          Defaults to None.
        posYZ (tuple, optional): Camera position when taking screenshot in YZ.
          Defaults to None.

    Returns:
        tuples: 3 camera position for the largest particle,
          used for future screenshots
    """
    p = pv.Plotter(off_screen=True)  # Initialize the camera
    p.add_mesh(polydata, color="black")  # Load the particle data
    p.view_xy()  # Move and rotate camera

    # This is done to ensure that camera is adjusted for
    # the largest particle in the dataset
    if posXY is not None:
        p.camera.position = posXY
    else:
        posXY = p.camera.position

    # Save generated picture in 64x64 resolution
    p.screenshot(f"Pictures/{date}/xy/P{P}_{C}_V{i}xy", window_size=[64, 64])

    p.view_xz()
    if posXZ is not None:
        p.camera.position = posXZ
    else:
        posXZ = p.camera.position
    p.screenshot(f"Pictures/{date}/xz/P{P}_{C}_V{i}xz", window_size=[64, 64])

    p.view_yz()
    if posYZ is not None:
        p.camera.position = posYZ
    else:
        posYZ = p.camera.position
    p.screenshot(f"Pictures/{date}/yz/P{P}_{C}_V{i}yz", window_size=[64, 64])

    return posXY, posXZ, posYZ


def gen_polydata(center_pos: list, radius: float):
    """Function generating a polydata based on center positions generated by function gen_centerpos

    Args:
        center_pos (list): list of center positions
        radius (float): radius of each sphere

    Returns:
        merged_spheres: pyvista polydata
    """
    # Radius of spheres
    radius = radius

    spheres = []
    for i in center_pos:  # Create a list of pyvista spheres
        spheres.append(pv.Sphere(center=i, radius=radius))

    # CleanToGrid, merge all spheres in list to a single object
    merged_spheres = pv.MultiBlock(spheres)\
        .combine().extract_surface().clean()
    merged_spheres.flip_normals()  # Needed for IBOFLOW to work correctly
    return merged_spheres


def gen_soot(num_particles: int, radius: float, domainData: list):
    """Function that generates n particle center points with, each particle is at a distance r or 2r (chosen at random) from each other

    :param num_particles: int - number of particles to be generated
    :param radius: float, radius of each sphere
    :return: center_pos, list - list of vectors in 3d with
        center position for each particle
    """
    #  Case data
    radius = radius
    center_pos = np.zeros((num_particles, 3))

    #  Set up mesh
    xT = domainData[3]
    yT = domainData[4]
    zT = domainData[5]

    delta_vec = [domainData[0], domainData[1], domainData[2]]

    Nx = xT / delta_vec[0]
    Ny = yT / delta_vec[1]
    Nz = zT / delta_vec[2]

    mesh = np.zeros((int(xT/radius), int(yT/radius), int(zT/radius)))

    max_n = np.shape(mesh)[0] * np.shape(mesh)[1] * np.shape(mesh)[2]
    if num_particles > max_n:
        raise ValueError("Too many particles requested")

    #  Place first sphere
    mesh[int(Nx/2), int(Ny/2), int(Nz/2)] = 1

    #  Set up vector of points where spheres are placed
    sphere_locations = [[int(Nx/2), int(Ny/2), int(Nz/2)]]

    #  Place spheres, slower version with while -> implement for loop
    #  instead with collision checking
    #  Always 1 radie overlap with
    while np.sum(mesh) < num_particles:
        #  Random direction to put next sphere (and step size r or 2r])
        direction = randrange(12)

        #  Random sphere to start from
        start_point = randrange(len(sphere_locations))
        index = sphere_locations[start_point]

        direction_changes = {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1),
                             3: (-1, 0, 0), 4: (0, -1, 0), 5: (0, 0, -1),
                             6: (2, 0, 0), 7: (0, 2, 0), 8: (0, 0, 2),
                             9: (-2, 0, 0), 10: (0, -2, 0), 11: (0, 0, -2)}

        if direction in direction_changes:
            change = direction_changes[direction]
            new_index = [index[i] + change[i] for i in range(3)]
            mesh[tuple(new_index)] = 1

        sphere_locations.append(new_index)

    #  This part would be unecessary with collision checking
    unique_locations = []
    for i in range(len(sphere_locations)):
        not_unique = 0
        location = sphere_locations[i]

        for j in range(i + 1, len(sphere_locations)):
            if location[0] == sphere_locations[j][0]:
                if location[1] == sphere_locations[j][1]:
                    if location[2] == sphere_locations[j][2]:
                        not_unique += 1

        if not_unique == 0:
            unique_locations.append(location)

    #  Mapping index to a coordinate
    for i in range(len(unique_locations)):
        index = unique_locations[i]

        center_pos[i, 0] = index[0] * delta_vec[0]
        center_pos[i, 1] = index[1] * delta_vec[1]
        center_pos[i, 2] = index[2] * delta_vec[2]

    return center_pos


def read_csv(filename: str):
    """Read data from csv file

    Args:
        filename (string): name of csv file to be read

    Returns:
        numpy array (nxm): matrix of all data read n columns
          of data with m data points each
    """
    df = pd.read_csv("data/" + filename)
    columns = df.columns.to_list()

    num_particles = np.array(df[columns[0]])

    # radius = np.array(df[columns[5]])
    AreaProj = np.array(df[columns[6]])
    Re = np.array(df[columns[7]])

    matrixInputData = np.array([num_particles, AreaProj, Re])
    C_d_vector = np.array(df[columns[8]])
    return matrixInputData, C_d_vector


def scale_data(matrix):
    """Scales data using sklearn standard scalers

    Args:
        matrix (list): list of lists that is to be scaled

    Returns:
        list: list of lists with scaled data
    """
    num_particle_scaler = StandardScaler()
    # radius_scaler = StandardScaler()
    Projected_area_scaler = StandardScaler()
    Reynolds_number_scaler = StandardScaler()

    num_particle = num_particle_scaler.fit_transform(matrix[0, :]
                                                     .reshape(-1, 1))
    # radius = radius_scaler.fit_transform(matrix[1, :].reshape(-1, 1))
    proj_area = Projected_area_scaler.fit_transform(matrix[1, :]
                                                    .reshape(-1, 1))
    reynold = Reynolds_number_scaler.fit_transform(matrix[2, :].reshape(-1, 1))

    return np.array([num_particle[:, 0], proj_area[:, 0], reynold[:, 0]])


def split_data(matrixTensor, C_d_vector_tensor, split: float):
    """Splitting data in to training and test data

    Args:
        matrixTensor (torch tensor dtype = float32): matrix consisting
          of n data columns with m data points each
        C_d_vector_tensor (1xm torch tensor dtype = float32): vector
        consisting of m data points for output variable C_d
        split (float): ratio of training to test

    Returns:
        X_train,X_test,Y_train,Y_test: Same data but
        split in to categories for training and testing
    """

    return train_test_split(matrixTensor, C_d_vector_tensor,
                            train_size=split, random_state=42)


def import_pictures(folder_name: str):
    """import the pictures to a suitable format for a CNN

    Args:
        folder_name (string): name of the folder where the pictures are stored

    Returns:
        numpy array: list of matrices for all pictures in the specified folder
    """
    data = []
    if isinstance(folder_name, str):
        files = os.listdir(folder_name)
        for file_name in files:
            image = imageio.imread(folder_name + '/' + file_name)
            data.append(np.reshape(image[:, :, 0]/255, (1, 64, 64)))
    else:
        files_1 = os.listdir(folder_name[0])
        files_2 = os.listdir(folder_name[1])
        files_3 = os.listdir(folder_name[2])
        for i in range(len(files_1)):
            image1 = imageio.imread(folder_name[0] + '/' + files_1[i])
            image2 = imageio.imread(folder_name[1] + '/' + files_2[i])
            image3 = imageio.imread(folder_name[2] + '/' + files_3[i])

            image_final = np.reshape((image1[:, :, 0]/255, image2[:, :, 0]/255,
                                      image3[:, :, 0]/255), (3, 64, 64))
            data.append(image_final)
    return data

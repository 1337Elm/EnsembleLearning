import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import imageio
from sklearn.model_selection import train_test_split


def read_csv(filename):
    """Read data from csv file

    Args:
        filename (string): name of csv file to be read

    Returns:
        numpy array (nxm): matrix of all data read n columns
          of data with m data points each
    """
    df = pd.read_csv("data/" + filename)
    # df = df[50::50]
    columns = df.columns.to_list()

    num_particles = np.array(df[columns[0]])

    forceX = np.array(df[columns[2]])
    forceY = np.array(df[columns[3]])
    forceZ = np.array(df[columns[4]])

    # radius = np.array(df[columns[5]])
    AreaProj = np.array(df[columns[6]])
    Re = np.array(df[columns[7]])

    matrixInputData = np.array([num_particles, AreaProj, Re])
    matrixOutputData = np.array([forceX, forceY, forceZ])
    C_d_vector = np.array(df[columns[8]])
    return matrixInputData, matrixOutputData, C_d_vector


def scale_data(matrix):
    """Scales data using sklearn standard scalers

    Args:
        matrix (list): list of lists that is to be scaled

    Returns:
        list: list of lists with scaled data
    """
    num_particle_scaler = StandardScaler()
    vel_scaler = StandardScaler()
    # radius_scaler = StandardScaler()
    Projected_area_scaler = StandardScaler()
    Reynolds_number_scaler = StandardScaler()

    num_particle = num_particle_scaler.fit_transform(matrix[0, :]
                                                     .reshape(-1, 1))
    vel = vel_scaler.fit_transform(matrix[1, :].reshape(-1, 1))
    # radius = radius_scaler.fit_transform(matrix[2, :].reshape(-1, 1))
    proj_area = Projected_area_scaler.fit_transform(matrix[3, :]
                                                    .reshape(-1, 1))
    reynold = Reynolds_number_scaler.fit_transform(matrix[4, :]
                                                   .reshape(-1, 1))

    return np.array([num_particle[:, 0], vel[:, 0],
                     proj_area[:, 0], reynold[:, 0]])


def split_data(matrixTensor, C_d_vector_tensor, split):
    """Splitting data in to training and test data

    Args:
        matrixTensor (torch tensor dtype = float32): matrix consisting of
          n data columns with m data points each
        C_d_vector_tensor (1xm torch tensor dtype = float32): vector consisting
          of m data points for output variable C_d
        split (float): ratio of training to test

    Returns:
        X_train,X_test,Y_train,Y_test: Same data but split in to categories
          for training and testing
    """

    return train_test_split(matrixTensor, C_d_vector_tensor,
                            train_size=split, random_state=42)


def import_pictures(folder_name):
    """import the pictures to a suitable format for a CNN

    Args:
        folder_name (string): name of the folder where the pictures are stored

    Returns:
        numpy array: list of matrices for all pictures in the specified folder
    """
    data = []
    files = os.listdir(folder_name)
    for file_name in files:
        image = imageio.imread(folder_name + '/' + file_name)
        data.append(np.reshape(image[:, :, 0]/255, (1, 64, 64)))
        # data.append(imageio.imread(folder_name + '/' + file_name)/256)

    return data

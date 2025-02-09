import torch
from utils import EnsembleModel, read_csv, import_pictures
import numpy as np


def exUsageEnsembleModel():
    """Example usage of Ensemble model, plug&play
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    folder_nameXY = "./Pictures/2024-04-03/xy/"
    folder_nameXZ = "./Pictures/2024-04-03/xz/"
    folder_nameYZ = "./Pictures/2024-04-03/yz/"
    filename_data = "data2024-04-03.csv"

    matrixInput, C_d_vector = read_csv(filename_data)

    nn_input = torch.tensor(matrixInput.transpose(),
                            dtype=torch.float32).to(device)

    data_AA = np.array(import_pictures([folder_nameXY,
                                        folder_nameXZ, folder_nameYZ]))

    cnn_input = torch.tensor(data_AA,
                             dtype=torch.float32).to(device)

    mnn = EnsembleModel([3, 10, 1], [64, 3], [2, 8, 1], device=device)
    mnn.load_model()

    # To predict a single value, specify input for cnn as i:i+1,
    # for the i:th data point
    print(mnn.predict(nn_input[1, :], cnn_input[1:2, :, :, :]))

    # Predict for all samples in the data
    print(mnn.predict(nn_input[:, :], cnn_input[:, :, :, :]))


if __name__ == "__main__":
    exUsageEnsembleModel()

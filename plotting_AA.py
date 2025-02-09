import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import torch


def showcase_model(model, data, analyt_sol, name, count, axes):
    """ Function that plots data from the model provided
        at subplot specified by count. It also plots the analytical solution.

    Args:
        model (object): class instance of a network which has been trained
        data (matrix): matrix containing all data needed to make predictions
          with the model
        analyt_sol (array): array of C_d values, ground truth
        name (str): name of model, used to differentiate the algorithm
          (CNN and DNN treated differently)
        count (int): index for which it is wished to plot the given model
        axes (array): array of axes used in the subplot figure
          ex, [ax1,ax2,ax3]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if name == "DNN":
        predictions = model(data).cpu().detach().numpy()
        Re = data.cpu().numpy().transpose()[2]

        ax = axes[count]

        ax.scatter(Re, predictions, marker='^', label="$C_{d-Pred}$", s=10)
        ax.scatter(Re, analyt_sol.cpu(), marker='o', facecolors='none',
                   edgecolors='k', label="$C_d$", s=10)
        ax.set_ylabel("$C_d$")
        ax.set_xlabel("Re")
        ax.legend()
        ax.title.set_text(name)
        ax.set_yscale("log")
        ax.set_xscale("log")

    elif name == "CNN":
        predictions = model(data[0]).cpu().detach().numpy()
        Re = data[1].cpu().numpy().transpose()[2]

        ax = axes[count]

        ax.scatter(Re, predictions, marker='^', label="Predicted Error", s=10)
        ax.scatter(Re, analyt_sol.cpu(), marker='o', facecolors='none',
                   edgecolors='k', label="Error from NN", s=10)
        ax.set_ylabel("Error")
        ax.set_xlabel("Re")
        ax.legend()
        ax.title.set_text(name)
        ax.set_yscale("log")
        ax.set_xscale("log")
    else:
        InDataNN = data[0]
        InDataCNN1 = data[1]

        y_pred1 = model[0](InDataNN).cpu().detach().numpy().squeeze()
        y_pred2 = model[1](InDataCNN1).cpu().detach().numpy()[:, 0]

        preds = torch.tensor(np.transpose(np.array([y_pred1, y_pred2])),
                             dtype=torch.float32).to(device)

        predictions = model[-1](preds).cpu().detach().numpy().squeeze()
        Re = InDataNN.cpu().numpy().transpose()[2]

        ax = axes[count]

        ax.scatter(Re, predictions, marker='^', label="$C_{d-Pred}$", s=10)
        ax.scatter(Re, analyt_sol.cpu(), marker='o', facecolors='none',
                   edgecolors='k', label="$C_d$", s=10)
        ax.set_ylabel("$C_d$")
        ax.set_xlabel("Re")
        ax.legend()
        ax.title.set_text(name)
        ax.set_yscale("log")
        ax.set_xscale("log")


def convergence_plot(loss_array, cnn_loss_array, mnn_loss_array):
    """ Function that creates a convergence plot for all models,
        currently however, only for DNN and MNN.

    Args:
        loss_array (array): array of calculated loss values for DNN network
        cnn_loss_array (array): array of calculated loss values for CNN network
        mnn_loss_array (array): array of calculated loss values for MNN network
    """
    plt.figure()
    plt.plot(loss_array, 'b', label="DNN")
    plt.plot(cnn_loss_array, 'r--', label="CNN1")
    plt.plot(mnn_loss_array, 'k-.', label="MNN")
    plt.ylim([0, 1])
    plt.xlim([15, np.max([len(mnn_loss_array), len(loss_array),
                          len(cnn_loss_array)]) + 2])
    plt.ylabel("$Loss$")
    plt.xlabel("$Epoch$")
    plt.legend()
    plt.title("Convergence Zoomed In")

    plt.savefig("./models/plots/convergence_NN_CNNAA_MNN_error_train.png",
                bbox_inches='tight')


def statistics(model, data, analyt_sol, model_name):
    """ Function that computes and prints statistical metrics
      using scikit-learn, for a specified model.

    Args:
        model (object): class instance of network which has been trained
        data (matrix): matrix containing all data needed
          to make predictions with specified model
        analyt_sol (array): array of C_d values, ground truth
        model_name (str): name of the model, MM treated differently,
          thus differentiating is needed
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("-----------------------" + f" Stats {model_name} "
          + "-------------------")
    if model_name == "NN" or model_name == "CNN":
        predictions = model(data).cpu().detach().numpy()
        print(f"RMSE of {model_name} model: \
               {rmse(predictions,analyt_sol.cpu())}")
        print(f"MAPE of {model_name} model: \
              {mean_absolute_percentage_error(predictions, analyt_sol.cpu())}")
        print(f"r-squared score of {model_name} model: \
              {r2_score(predictions, analyt_sol.cpu())}")
    else:
        InDataNN = data[0]
        InDataCNN1 = data[1]

        y_pred1 = model[0](InDataNN).cpu().detach().numpy().squeeze()
        y_pred2 = model[1](InDataCNN1).cpu().detach().numpy()[:, 0]
        preds = torch.tensor(np.transpose(np.array([y_pred1, y_pred2])),
                             dtype=torch.float32).to(device)

        predictions = model[-1](preds).cpu().detach().numpy()
        print(f"RMSE of {model_name} model: \
               {rmse(predictions,analyt_sol.cpu())}")
        print(f"MAPE of {model_name} model: \
              {mean_absolute_percentage_error(predictions, analyt_sol.cpu())}")
        print(f"r-squared score of {model_name} model: \
              {r2_score(predictions, analyt_sol.cpu())}")

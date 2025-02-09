import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import torch
import pandas as pd


def showcase_model(model: torch.nn.module, data: pd.DataFrame,
                   analyt_sol: np.ndarray, name: str, count, axes):
    """_summary_

    Args:
        model (torch.nn.module): Model to be showcased
        data (pd.DataFrame): Data to test model on
        analyt_sol (np.ndarray): Analytical solution to compare model with
        name (str): name of model
        count (_type_): _description_
        axes (_type_): _description_
    """
    if name == "NN":
        predictions = model(data).detach().numpy()
        Re = data.numpy().transpose()[3]

        ax = axes[count]

        ax.scatter(Re, predictions, marker='^', label="$C_{d-Pred}$", s=10)
        ax.scatter(Re, analyt_sol, marker='o', facecolors='none',
                   edgecolors='k', label="$C_d$", s=10)
        ax.set_ylabel("$C_d$")
        ax.set_xlabel("Re")
        ax.legend()
        ax.title.set_text(name)
        ax.set_yscale("log")
        ax.set_xscale("log")

    elif name == "CNN":
        predictions = model(data[0]).detach().numpy()
        Re = data[1].numpy().transpose()[3]

        ax = axes[count]

        ax.scatter(Re, predictions, marker='^', label="$C_{d-Pred}$", s=10)
        ax.scatter(Re, analyt_sol, marker='o', facecolors='none',
                   edgecolors='k', label="$C_d$", s=10)
        ax.set_ylabel("$C_d$")
        ax.set_xlabel("Re")
        ax.legend()
        ax.title.set_text(name + str(count))
        ax.set_yscale("log")
        ax.set_xscale("log")
    else:
        InDataNN = data[0]
        InDataCNN1 = data[1]
        InDataCNN2 = data[2]
        InDataCNN3 = data[3]

        y_pred1 = model[0](InDataNN).detach().numpy().squeeze()
        y_pred2 = model[1](InDataCNN1).detach().numpy()[:, 0]
        y_pred3 = model[2](InDataCNN2).detach().numpy()[:, 0]
        y_pred4 = model[3](InDataCNN3).detach().numpy()[:, 0]
        preds = np.transpose(torch.tensor(np.array([y_pred1, y_pred2,
                                                    y_pred3, y_pred4]),
                                          dtype=torch.float32))

        predictions = model[-1](preds).detach().numpy().squeeze()
        Re = InDataNN.numpy().transpose()[3]

        ax = axes[count]

        ax.scatter(Re, predictions, marker='^', label="$C_{d-Pred}$", s=10)
        ax.scatter(Re, analyt_sol, marker='o', facecolors='none',
                   edgecolors='k', label="$C_d$", s=10)
        ax.set_ylabel("$C_d$")
        ax.set_xlabel("Re")
        ax.legend()
        ax.title.set_text(name)
        ax.set_yscale("log")
        ax.set_xscale("log")


def convergence_plot(loss_array: np.ndarray, cnn_loss_array: np.ndarray,
                     cnn_loss_array2: np.ndarray, cnn_loss_array3: np.ndarray,
                     mnn_loss_array: np.ndarray):
    """_summary_

    Args:
        loss_array (np.ndarray): _description_
        cnn_loss_array (np.ndarray): _description_
        cnn_loss_array2 (np.ndarray): _description_
        cnn_loss_array3 (np.ndarray): _description_
        mnn_loss_array (np.ndarray): _description_
    """
    plt.figure()
    plt.plot(loss_array, 'b', label="NN")
    plt.plot(cnn_loss_array, 'r--', label="CNN1")
    plt.plot(cnn_loss_array2, 'r-.', label="CNN2")
    plt.plot(cnn_loss_array3, 'r:', label="CNN3")
    plt.plot(mnn_loss_array, 'k-.', label="MNN")
    plt.ylim([0, 1])
    plt.ylabel("$Loss$")
    plt.xlabel("$Epoch$")
    plt.legend()
    plt.title("Convergence")

    plt.savefig("./models/plots/convergence.png", bbox_inches='tight')


def statistics(model, data, analyt_sol, model_name):
    print("-----------------------" + f" Stats {model_name} " +
          "-------------------")
    if model_name == "NN" or model_name == "CNN":
        predictions = model(data).detach().numpy()
        print(f"RMSE of {model_name} model: {rmse(predictions, analyt_sol)}")
        print(f"MAPE of {model_name} model: \
              {mean_absolute_percentage_error(predictions, analyt_sol)}")
        print(f"r-squared score of {model_name} model: \
              {r2_score(predictions, analyt_sol)}")
    else:
        InDataNN = data[0]
        InDataCNN1 = data[1]
        InDataCNN2 = data[2]
        InDataCNN3 = data[3]

        y_pred1 = model[0](InDataNN).detach().numpy().squeeze()
        y_pred2 = model[1](InDataCNN1).detach().numpy()[:, 0]
        y_pred3 = model[2](InDataCNN2).detach().numpy()[:, 0]
        y_pred4 = model[3](InDataCNN3).detach().numpy()[:, 0]
        preds = np.transpose(torch.tensor(np.array([y_pred1, y_pred2,
                                                    y_pred3, y_pred4]),
                                          dtype=torch.float32))

        predictions = model[-1](preds).detach().numpy()
        print(f"RMSE of {model_name} model: {rmse(predictions, analyt_sol)}")
        print(f"MAPE of {model_name} model: \
              {mean_absolute_percentage_error(predictions, analyt_sol)}")
        print(f"r-squared score of {model_name} model: \
              {r2_score(predictions, analyt_sol)}")

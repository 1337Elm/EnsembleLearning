# ----------------------- Import packages -------------------------#
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error as RMS
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score as R2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def SVR_main():
    """ This function creates, fits and evaluates an SVR model that predict drag coefficient
        using the projected area of the particle cluster in the flow direction and the reynolds number
    """
    # ----------------------- load data -------------------------#
    # Change data set here
    data_path = "data/data2024-04-03.csv"
    df = pd.read_csv(data_path, index_col=None)

    fx = df["C_d"]

    # Feutures will contain velocity, projected area, reynolds number and C_d
    feutures = df.drop(columns=["forces_x", "radius", "forces_y", "forces_z",
                                "num_particles", "velocity"])

    # ----------------------- Pre-Processing -------------------------#
    # One scaler per feuture is the recommended practice

    # Random_state ensure that is splits the same each time
    X_train, X_test, y_train, y_test = train_test_split(feutures, fx,
                                                        test_size=0.20,
                                                        random_state=42)

    scaler = StandardScaler
    projected_area_scaler = scaler()
    Reynolds_number_scaler = scaler()
    fx_scaler = scaler()
    r = X_train.shape[0]


    projected_area_scaled = projected_area_scaler.fit_transform(X_train["yzarea"]
                                                                .to_numpy()
                                                                .reshape(-1, 1))
    Reynolds_number_scaled = Reynolds_number_scaler.fit_transform(
        X_train["Reynolds number"].to_numpy().reshape(-1, 1))

    y_train_ready = fx_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))

    number_of_feutures = 2
    X_train_ready = np.zeros((len(Reynolds_number_scaled), number_of_feutures))
    X_train_ready[:, 0] = projected_area_scaled[:, 0]
    X_train_ready[:, 1] = Reynolds_number_scaled[:, 0]

    # ----------- Testing data -----------#
    # Extract each feauture, convert to numpy and reshape to a 2d column
    # Should not fit scaler here, only transform
    projected_area_test = projected_area_scaler.transform(X_test["yzarea"]
                                                          .to_numpy()
                                                          .reshape(-1, 1))
    Reynolds_number_test = Reynolds_number_scaler.transform(
        X_test["Reynolds number"].to_numpy().reshape(-1, 1))

    X_test_ready = np.zeros((len(Reynolds_number_test), number_of_feutures))
    X_test_ready[:, 0] = projected_area_test[:, 0]
    X_test_ready[:, 1] = Reynolds_number_test[:, 0]

    # ----------------------- Setup SVR -------------------------#
    # HyperParameters
    Cp = 100
    eps = 1e-5

    # Select Model and kernel
    model = SVR(kernel='rbf', C=Cp, epsilon=eps, cache_size=2000, tol=1e-3)
    fitted_model = model.fit(X_train_ready, y_train_ready.ravel())

    fx_predict = fx_scaler.inverse_transform(fitted_model.predict(X_test_ready)
                                             .reshape(-1, 1))

    # ----------------------- Evaluation -------------------------#

    y_test = y_test.to_numpy()
    RMS_error = RMS(y_test, fx_predict)
    R2_score = R2(y_test, fx_predict)
    MAP_error = MAPE(y_test, fx_predict) * 100

    print("-----------------------------------\n")
    print(f"The mode gave an RMS-error of {RMS_error}")
    print("-----------------------------------\n")
    print(f"The mode gave an MAP-error of {MAP_error}")
    print("-----------------------------------\n")
    print(f"The mode gave an R2-score of {R2_score}")
    print("-----------------------------------\n")

    plt.figure(1)
    plt.title("Prediction from SVR")
    plt.scatter(X_test["Reynolds number"], fx_predict.flatten(), s=10)
    plt.scatter(X_test["Reynolds number"], y_test.flatten(), s=10)
    plt.legend(["$C_{d-Pred}$", "$C_d$"])
    plt.ylabel("$C_d$")
    plt.xlabel("Re")
    plt.yscale("log")
    plt.xscale("log")

    plt.figure(2)
    plt.title("Prediction vs True value")
    plt.scatter(X_test["yzarea"], y_test.flatten(), s=10)
    plt.scatter(X_test["yzarea"], fx_predict.flatten(), s=10)
    plt.legend(["Cd", "Cd_pred"])
    plt.ylabel("Cd")
    plt.xlabel("Yz-Area")

    plt.figure(3)
    plt.title("Feuture Correlation")
    corrmatrix = df.drop(columns=["radius", "forces_y",
                                  "forces_z"]).corr(method="spearman")
    sns.heatmap(corrmatrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    Re, yz = X_test["Reynolds number"], X_test["yzarea"]
    Cd_pred = fx_predict.flatten()
    Cd_test = y_test.flatten()
    ax.scatter(Re, yz, MAP_error, c='r', label="Mean Average percentage Error")
    # ax.scatter(Re, yz, Cd_test, c='g')
    plt.title("Illustrated MAPE for SVR")
    ax.set_ylabel("YZ-Area")
    ax.set_xlabel("Reynolds Number")
    ax.legend()

    plt.show()


if __name__ == '__main__':
    SVR_main()

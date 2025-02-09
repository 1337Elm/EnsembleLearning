import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv():
    """Reads csv file, containing data needed to plot drag curve, using pandas.

    Returns:
        dataframe: pandas dataframe
    """
    df = pd.read_csv("dataForceRefined.csv")
    return df


def main():
    """ Code that plots a drag curve analytically as well as
        from the simulated data to see where the simulated data is accurat.
    """
    # case data
    density = 1
    r = 0.5e-3
    viscosity = 1.0e-5

    df = read_csv()
    forceX = df.ForceX
    velocityX = df.VelocityX

    c_drag = np.zeros(len(forceX))
    Re_d = np.zeros(len(forceX))
    c_drag_analyt = np.zeros(len(forceX))

    for i in range(len(forceX)):
        c_drag[i] = forceX[i] / (0.5*density*(velocityX[i]**2) *
                                 (np.pi * (r**2)))
        Re_d[i] = density*velocityX[i] * (2 * r)/viscosity

        c_drag_analyt[i] = 24 / Re_d[i]

    plt.plot(Re_d, c_drag, label="Simulated")
    plt.plot(Re_d, c_drag_analyt, label="Analytical  $C_d = 24/Re_d$")
    plt.xlabel("$Re_d$")
    plt.ylabel("$C_D$")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.title("Drag curve for sphere")
    plt.savefig("figures/drag_plot_from_dataframeRefined")
    plt.show()


if __name__ == '__main__':
    main()

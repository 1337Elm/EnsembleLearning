from utils import gen_soot, gen_polydata
from sphere_3d import run_simulation
from utils import picture_generator
import pandas as pd
import numpy as np
import os
import shapely
import time


def Generate_data():
    """
    Function that generates a dataset containing information about a
    fluid simulation done using IBOFLOW.
    """
    # Make dirs
    date = time.strftime("%Y-%m-%d")
    os.makedirs(f"vtkfiles/{date}", exist_ok=True)
    os.makedirs(f"Pictures/{date}/xy", exist_ok=True)
    os.makedirs(f"Pictures/{date}/xz", exist_ok=True)
    os.makedirs(f"Pictures/{date}/yz", exist_ok=True)
    os.makedirs(f"xmffiles/{date}", exist_ok=True)

    # Changing radius does currently not work as expected
    # [m] Positive floats, must match sphere_3d.py
    # [m^2/s] Kinematic viscosity of fluid in simulation,
    #  must match sphere_3d.py
    # [kg/m3] Density of fluid in simulation
    radius = 0.5e-3
    viscosity = 1.0e-5
    rho = 1

    # Changes the domain size:
    # Current setup ensures 20 radius distance to walls
    #  with particles up to 15 subspheres
    # delta_vec[0],delta_vec[1],delta_vec[2],xT,yT,zT ~Domain information
    domainData = [radius, radius, radius,
                  4*(radius + 2*14*radius + 20*2*radius),
                  2*(radius + 2*14*radius + 20*2*radius),
                  2*(radius + 2*14*radius + 20*2*radius)]

    # Keep 2:1:1 ratio?20*15*2*radius, 20*15*radius, 20*15*radius

    # Variables below are free to change
    # list with Number of subspheres in each particle, going from high to low.
    # list with different fluid velocities to run
    # each generated particle with. Recommended interval: Low: 1e-3, High. 1
    num_particles = np.linspace(15, 1, 15, dtype=int)
    vel = np.linspace(1e-3, 5e-1, 40)
    ref = np.linspace(1, 4, 4, dtype=int)

    # Create a dataframe.
    df = pd.DataFrame({"num_particles": [], "velocity": [],
                       "forces_x": [], "forces_y": [], "forces_z": [],
                       "radius": [], "yzarea": [], "Reynolds number": [],
                       "C_d": [], "Refinement": []})
    df.to_csv(f"data/data{date}.csv", index=False, header=True)

    # Loop over list that specifies number of sub-spheres
    counter = 0
    for P in num_particles:
        # Configuration
        C = 0

        # Generate centerpoints of each subsphere
        # Create Polydata for the particle
        formation = gen_soot(P, radius, domainData)
        geometry = gen_polydata(formation, radius)

        # Calculate projected area for yz plane
        origin = geometry.center
        origin[1] += (P + 1) * (2 * radius)
        projection = geometry.project_points_to_plane(origin, normal=[1, 0, 0])

        merged = shapely.union_all([shapely.Polygon(projection.points[tri, 1:])
                                    for tri in projection.triangulate()
                                    .regular_faces])
        yzarea = merged.area

        # Calculate the characteristic length of particle
        L_ch = np.sqrt(yzarea)

        # Saves a vtk file that can be examined in paraviewer
        geometry.save(f"vtkfiles/{date}/P{P}_{C}.vtk")

        # Loop over list that specifies fluid velocity
        for i, v in enumerate(vel):

            for ref_i in ref:
                counter += 1
                # Not optimal error handling
                try:
                    # IBOFLOW calculates the resulting force on the particle
                    force = run_simulation(geometry, radius, domainData, v,
                                           ref_i, counter, date)
                except Exception as e:
                    print(e)
                    continue

                # Calculate the reynolds number
                # Calculates the coefficient of drag using the resulting force
                Re = v * L_ch * rho / viscosity
                C_d = force[0]/(0.5 * rho * v**2 * yzarea)

                new_row = pd.DataFrame({"num_particles": [P], "velocity": [v],
                                        "forces_x": [force[0]],
                                        "forces_y": [force[1]],
                                        "forces_z": [force[2]],
                                        "radius": [radius],
                                        "yzarea": [yzarea],
                                        "Reynolds number": [Re],
                                        "C_d": [C_d], "Refinement": [ref_i]})

                df = pd.concat([df, new_row])
                df.to_csv(f"data/data{date}.csv", index=False, header=True)

                # Generate pictures for each particle
                # (and velocity, not optimal)
                C += 1
                if P == np.max(num_particles):
                    posXY, posXZ, posYZ = picture_generator(P, C, i,
                                                            geometry, date)
                else:
                    picture_generator(P, C, i, geometry, date,
                                      posXY, posXZ, posYZ)


if __name__ == '__main__':
    Generate_data()

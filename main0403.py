from utils import gen_soot
from utils import gen_polydata
from sphere_3d import run_simulation
from utils import picture_generator
import pandas as pd
import numpy as np
import os
import shapely
import time

# TODO: Fix radius domain bug


def main():
    """
    Function that first generates a particle, (stores it),
    generates a mesh for the combined object of all spheres
      and finally runs a fluid simulation on it
    """
    # Make dirs
    date = time.strftime("%Y-%m-%d")
    os.makedirs(f"vtkfiles/{date}", exist_ok=True)
    os.makedirs(f"Pictures/{date}/xy", exist_ok=True)
    os.makedirs(f"Pictures/{date}/xz", exist_ok=True)
    os.makedirs(f"Pictures/{date}/yz", exist_ok=True)

    # Changing radius does currently not work as expected
    radius = 0.5e-3  # Positive floats
    viscosity = 1.0e-5
    rho = 1
    P = 0

    # Do not change:
    # delta_vec[0], delta_vec[1], delta_vec[2], xT, yT, zT ~Domain information
    domainData = [radius, radius, radius, 0.02, 0.01, 0.01]

    # Variables below are free to change
    num_particles = np.linspace(1, 15, 15, dtype=int)  # Positve integers
    vel = np.linspace(1e-2, 1, 50)  # floats

    df = pd.DataFrame({"num_particles": [], "velocity": [],
                       "forces_x": [], "forces_y": [], "forces_z": [],
                       "radius": [], "yzarea": [], "Reynolds number": [],
                       "C_d": []})
    df.to_csv(f"data/data{date}.csv", index=False, header=True)

    for particles in num_particles:
        P = particles
        C = 0

        formation = gen_soot(particles, radius, domainData)
        geometry = gen_polydata(formation, radius)

        # Calculate projected area for yz plane
        origin = geometry.center
        origin[1] += (particles + 1) * (2 * radius)
        projection = geometry.project_points_to_plane(origin, normal=[1, 0, 0])

        merged = shapely.union_all([shapely.Polygon(projection.points[tri, 1:])
                                    for tri in projection.triangulate()
                                    .regular_faces])
        yzarea = merged.area

        L_ch = np.sqrt(yzarea)

        geometry.save(f"vtkfiles/{date}/P{P}_{C}.vtk")
        for i, v in enumerate(vel):

            # Not optimal error handling
            try:
                force = run_simulation(geometry, radius, domainData, v)
            except Exception as e:
                print(e)
                continue

            Re = v * L_ch * rho / viscosity
            C_d = force[0]/(0.5 * rho * v**2 * yzarea)
            new_row = pd.DataFrame({"num_particles": [particles],
                                    "velocity": [v], "forces_x": [force[0]],
                                    "forces_y": [force[1]],
                                    "forces_z": [force[2]],
                                    "radius": [radius], "yzarea": [yzarea],
                                    "Reynolds number": [Re], "C_d": [C_d]})

            df = pd.concat([df, new_row])
            df.to_csv(f"data/data{date}.csv", index=False, header=True)

            # Use below to generate pictures for each particle
            C += 1
            # Picture generator has been updated since 0403, see other main
            picture_generator(P, C, i, geometry, date)


if __name__ == '__main__':
    main()

'''
IBOFlow-Sphere3D
'''

import iboflow.boundarycondition as ib_bc
import iboflow.interfaceoptions as ib_ifopt
import iboflow.materials as ib_m
import iboflow.objects as ib_o
import iboflow.refinements as ib_r
import iboflow.simulationdomain as ib_sim
import iboflow.solverinterface as ib_si


def create_iboflow_solver(PolyData, radius: float, domainData: list,
                          vel: float, ref: int, date: str):
    """Creates the fluid solver and specifies, domain, fluid and particle data

    Args:
        PolyData (Pyvista PolyData): Particle geometry data
        radius (float): radius of each sphere [m]
        domainData (list): list that specifies the domain
        vel (float): Fluid velocity [m/s]
        ref: (int): Refinement level for grid size near particle cluster
        date (string): Date for saving xmf files in correct folder

    Returns:
        Class: iboflowsolver
    """

    # Solver Options
    solver_opt = ib_ifopt.SolverOptions()  # Create solver option object
    solver_opt.fluid_solver_type = ib_ifopt.FluidSolverType.STEADY_STATE
    solver_opt.max_simple_crit = 1.0e-3  # Stopping criteria
    solver_opt.max_solver_it = 20000  # Maximum numbers of iterations
    solver_opt.num_openmp_threads = 8  # Number of threads to run he solver on

    io_opt = ib_ifopt.InputOutputOptions()
    io_opt.case_name = "ParticleSimulation"
    io_opt.working_directory = f"xmffiles/{date}"
    io_opt.time_write = 20
    io_opt.output_level = ib_ifopt.OutputLevel.STANDARD
    io_opt.output_file_type = ib_ifopt.OutputFileType.XMF

    # Immersed boundaries
    xB = 0
    yB = 0
    zB = 0

    xT = domainData[3]
    yT = domainData[4]
    zT = domainData[5]

    delta_vec = [domainData[0], domainData[1], domainData[2]]

    # Boundary conditions
    velocity_inlet = ib_bc.VelocityInlet([vel, 0, 0])
    fluid_inlet = ib_bc.FluidInletBoundaryCondition(inlet_type=velocity_inlet)
    inlet = ib_bc.BoundaryCondition(fluid_inlet, None, None, None)

    pressure_outlet = 0.0
    fluid_outlet = ib_bc.FluidStaticPressureOutletBoundaryCondition(
        static_outlet_pressure=pressure_outlet)
    outlet = ib_bc.BoundaryCondition(fluid_outlet, None, None, None)

    fluid_symmetry = ib_bc.FluidSymmetryBoundaryCondition()
    symmetry = ib_bc.BoundaryCondition(fluid_symmetry, None, None, None)

    # Assign boundaries
    boundaries = []

    bc_domain_1 = ib_bc.DomainSide(ib_bc.DomainSide.X_BOTTOM)
    boundaries += [(bc_domain_1, inlet)]

    bc_domain_2 = ib_bc.DomainSide(ib_bc.DomainSide.X_TOP)
    boundaries += [(bc_domain_2, outlet)]

    bc_domain_3 = ib_bc.DomainSide(ib_bc.DomainSide.Y_BOTTOM)
    boundaries += [(bc_domain_3, symmetry)]

    bc_domain_4 = ib_bc.DomainSide(ib_bc.DomainSide.Y_TOP)
    boundaries += [(bc_domain_4, symmetry)]

    bc_domain_5 = ib_bc.DomainSide(ib_bc.DomainSide.Z_BOTTOM)
    boundaries += [(bc_domain_5, symmetry)]

    bc_domain_6 = ib_bc.DomainSide(ib_bc.DomainSide.Z_TOP)
    boundaries += [(bc_domain_6, symmetry)]

    # Sphere object
    # Material
    sphere_density = 2700.0
    material = ib_m.SolidMaterial(sphere_density)

    # Mesh representation
    radius = radius
    sphere_mesh = PolyData

    vertices = sphere_mesh.points.tolist()
    connectivity = sphere_mesh.faces.tolist()
    triangles = []
    for i in range(1, len(connectivity), 4):
        triangles.append(connectivity[i:i + 3])

    mesh_representation = ib_o.SurfaceMeshData(vertices, triangles)

    # Refinement, makes the mesh more fine near the particle
    ref_level = ref
    ref_length = 3.0 * radius
    narrow_channel = ib_r.NarrowChannel(max_level=0,
                                        max_width=0.0,
                                        min_width=0.0,
                                        min_cells=0,
                                        padding=0.0)
    obj_ref_fluid = ib_r.ObjectRefinement(ib_r.Grid.FLUID, ref_level,
                                          ref_length, narrow_channel)

    # Boundary condition
    boundary_velocity = [0.0, 0.0, 0.0]
    bc_solid = ib_bc.BoundaryCondition(
        ib_bc.FluidWallBoundaryCondition(boundary_velocity), None, None, None)

    # Create object
    sphere = ib_o.SolidObject(obj_ref_fluid, None, mesh_representation,
                              bc_solid, material)

    # Fluid object
    fluid_density = 1.0
    fluid_viscosity = 1.0e-5  # Dynamic
    rheology = ib_m.ConstantRheology(fluid_viscosity)
    fluid = ib_m.FluidMaterial(fluid_density, rheology)

    # Setting up solver and viewer
    lin_alg = ib_ifopt.LinearAlgebraOptions()
    lin_alg.momentum_solver = ib_ifopt.LinearAlgebraSolver.IDR_JACOBI_GPU
    lin_alg.pressure_solver = ib_ifopt.LinearAlgebraSolver.CG_AMG_GPU
    gravity = [0.0, -9.82, 0.0]
    iboflow_options = ib_ifopt.Options(solver_opt, io_opt,
                                       lin_alg, gravity)

    lower_corner = [xB, yB, zB]
    upper_corner = [xT, yT, zT]
    base_cell_size = delta_vec
    fluid_domain = ib_sim.SimulationDomain(boundaries, fluid, lower_corner,
                                           upper_corner, base_cell_size)

    iboflow_solver = ib_si.SolverInterface(iboflow_options, fluid_domain)

    iboflow_solver.add_object(sphere)
    return iboflow_solver


def run_simulation(polyData, radius: float, domainData: list,
                   vel: float, ref: int, index: int, date: str):
    """Main function for running the simulation and returning the resulting force

    Args:
        polyData (PyVista PolyData): Object describing the particle geometry
        radius (float): [m] radius of each sub-sphere in the particle
        domainData (list): list of domain data
        vel (float): [m/s] Fluid velocity
        ref: (int): Refinement level for grid size near particle cluster
        index (): For file naming
        date (string): Date for saving xmf files in correct folder


    Returns:
        array: resulting forces in from three direction, wheras first is in flow direction.
    """
    solver = create_iboflow_solver(polyData, radius, domainData,
                                   vel, ref, date)
    solution_viewer = solver.get_solution_viewer()
    solver.simulate_to_steady_state()
    force = solution_viewer.get_fluid_force(0)
    solver.write_solution_to_disc(index)

    return force

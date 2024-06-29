This code uses cylindrical coordinates. It is meant to simulate blood vessels.
To use, run fluid_dynamics_sim.py

# Set your initial conditions and export frequency:
cylinder = fs.Cylinder(radius=1.0, length=10.0, num_r=50, num_theta=50, num_z=50)
fluid = fs.Fluid(rho=1.225, nu=1.81e-5)  # Air at sea level
CFL = 0.5
iterations = 100
VTK_interval = 10
    
# Set your boundary conditions
vInflow, vOutflow = 1, 1
pInflow = 1

The output will be .txt files containing velocities in the r, theta and z directions, 
pressure, and .vtk files to open in ParaView or your visualisation toolkit of choice.

To add obstacles such as an occlusion, edit the cylinder dimensions and add void spaces.
Add the new boundary conditions as well.
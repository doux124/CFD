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

In ParaView:
Select the Data Representation.
In the Pipeline Browser, select the loaded data.
In the Properties panel, go to the Representation dropdown and choose 3D Glyphs or Surface with Edges.

Apply a Glyph Filter.
Go to Filters > Alphabetical > Glyph.
In the Properties panel of the Glyph filter, set Glyph Type to Arrow.
Set Scalars to None and Vectors to Velocity.

Adjust Glyph Parameters.
Set the Scale Factor to adjust the size of the arrows representing the velocity vectors.
Click Apply to see the glyphs.
Create a 3D Heat Map of Velocity Magnitude

Add a Calculator Filter.
Go to Filters > Alphabetical > Calculator.
In the Properties panel of the Calculator, set Result Array Name to VelocityMagnitude.
In the Expression field, enter mag(Velocity) to calculate the magnitude of the velocity vectors.
Click Apply.

Color by Velocity Magnitude.
In the Pipeline Browser, select the Calculator filter.
In the Properties panel, set the Coloring dropdown to VelocityMagnitude.

Adjust Color Map.
Click on the color bar to the right of the Render View to open the Color Map Editor.
Adjust the color map to highlight different ranges of velocity magnitudes.

Streamlines (Optional)
Add a Stream Tracer Filter.
Go to Filters > Alphabetical > Stream Tracer.
In the Properties panel, set the Input Vectors to Velocity.
Adjust Seed Type, Resolution, and other parameters to control the streamlines' appearance.
Click Apply.

To add obstacles such as an occlusion, edit the cylinder dimensions and add void spaces.
Add the new boundary conditions as well.
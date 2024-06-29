"""
Written by Jordan Low Jun Yi; NUS BME Y0
Created on 6/28/2024
Hours spent: 12

Read README for usage details
"""

import numpy as np
import os
import vtk

np.seterr(divide='ignore')

"""
Cylinder functions:
createSource(source)
setVelocityBoundaries(inflow v, outflow v)
setPressureBoundaries(inflow p)

Library functions;
starred_velocities(cylinder, fluid, dt)
solve_pressure_poisson(u_r_star, u_z_star, cylinder, fluid, dt)

laplacian_cylindrical(field, dr, dtheta, dz)
d1(field, dr, dtheta, dz): # first derivative
d2(field, dr, dtheta, dz): # second derivative

makeResultDirectories(wipe=False)
writeToVTK(cylinder, i, interval)
"""

class Cylinder:
    def __init__(self, radius, length, num_r, num_theta, num_z):
        self.radius = radius
        self.length = length
        self.num_r = num_r
        self.num_theta = num_theta
        self.num_z = num_z
        self.radial_points = np.linspace(0, self.radius, self.num_r)
        self.grid = self.createGrid(self.radial_points)
        self.createSource()
        self.createMatrices()
        self.setDeltas()

    def createGrid(self, radial_points):
        axial_points = np.linspace(0, self.length, self.num_z)
        r, theta, z = np.meshgrid(radial_points, np.linspace(0, 2 * np.pi, self.num_theta), axial_points)
        return r, theta, z

    def createSource(self, source=None):
        if source is None:
            source = np.zeros_like(self.grid[0])
        self.source = source

    def createMatrices(self):
        self.u = np.zeros_like(self.grid[0])
        self.v = np.zeros_like(self.grid[0])
        self.w = np.zeros_like(self.grid[0])
        self.p = np.zeros_like(self.grid[0])
    
    def setDeltas(self):
        self.dr = self.radius / (self.num_r - 1)
        self.dtheta = 2 * np.pi / (self.num_theta - 1)
        self.dz = self.length / (self.num_z - 1)
        
    def setVelocityBoundaries(self, u_inflow, u_outflow):
        # Left boundary (inflow)
        self.u[:, 0, :] = u_inflow
        
        # Right boundary (outflow)
        self.u[:, -1, :] = u_outflow
        
        # Azimuthal boundaries (no-slip)
        self.v[:, :, 0] = 0
        self.v[:, :, -1] = 0
        
        # Axial boundaries (no-slip)
        self.w[:, 0, :] = 0
        self.w[:, -1, :] = 0

        # Set u, v, w at radial boundaries (no-slip for simplicity)
        self.u[0, :, :] = 0
        self.u[-1, :, :] = 0

    def setPressureBoundaries(self, p_inflow):  
        # Left boundary (inflow)
        self.p[:, 0, :] = p_inflow

    def display_geometry(self):
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(self.radius)
        cylinder.SetHeight(self.length)
        cylinder.SetResolution(self.num_radial_points)

        cylinderMapper = vtk.vtkPolyDataMapper()
        cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

        cylinderActor = vtk.vtkActor()
        cylinderActor.SetMapper(cylinderMapper)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(cylinderActor)
        renderer.SetBackground(0.1, 0.2, 0.4)

        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        style = vtk.vtkInteractorStyleTrackballCamera()
        renderWindowInteractor.SetInteractorStyle(style)

        renderWindow.Render()
        renderWindowInteractor.Start()

class Fluid: 
    def __init__(self, rho, nu):
        self.rho = rho
        self.nu = nu
    
def timeStep(CFL, cylinder):
    max_velocity = np.max(np.sqrt(cylinder.u**2 + cylinder.v**2 + cylinder.w**2))
    dx = min(cylinder.dr, cylinder.dtheta, cylinder.dz)
    dt = CFL * dx / max_velocity
    if dt > 100 or np.isinf(dt): # for values approaching inf
        dt = cylinder.dr + cylinder.dtheta + cylinder.dz
        print("inf:", dt)
    return dt

def solve_navier_stokes(CFL, cylinder, fluid):
    u, v, w, p, rho, nu, dr, dtheta, dz = cylinder.u, cylinder.v, cylinder.w, cylinder.p, fluid.rho, fluid.nu, cylinder.dr, cylinder.dtheta, cylinder.dz
    dt = timeStep(CFL, cylinder)
    grid = cylinder.grid
    p = solve_pressure_poisson(p, rho, dr, dtheta, dz, dt, u, v, w)

    nr, ntheta, nz = u.shape  # Extract grid dimensions

    # Velocity gradients
    du_dr = np.gradient(u, axis=0) / dr
    dv_dtheta = np.gradient(v, axis=1) / dtheta
    dw_dz = np.gradient(w, axis=2) / dz

    # Non-linear convection terms
    convective_u = u * du_dr + (v / np.expand_dims(np.arange(ntheta), axis=0)) * dv_dtheta
    convective_v = u * du_dr + (v / np.expand_dims(np.arange(ntheta), axis=0)) * dv_dtheta
    convective_w = u * du_dr + (v / np.expand_dims(np.arange(ntheta), axis=0)) * dv_dtheta

    # Diffusion terms
    laplacian_u = laplacian_cylindrical(u, dr, dtheta, dz)
    laplacian_v = laplacian_cylindrical(v, dr, dtheta, dz)
    laplacian_w = laplacian_cylindrical(w, dr, dtheta, dz)

    # Pressure gradient
    dp_dr = np.gradient(p, axis=0) / dr
    dp_dtheta = np.gradient(p, axis=1) / dtheta
    dp_dz = np.gradient(p, axis=2) / dz

    # Update velocities
    u += dt * (-convective_u - dp_dr / rho + nu * laplacian_u)
    v += dt * (-convective_v - dp_dtheta / (rho * np.expand_dims(np.arange(ntheta), axis=0)) + nu * laplacian_v)
    w += dt * (-convective_w - dp_dz / rho + nu * laplacian_w)

    cylinder.u = u
    cylinder.v = v
    cylinder.w = w
    cylinder.p = p

    return cylinder


def solve_pressure_poisson(u_r, u_z, cylinder, fluid, dt, max_iter=1000, tol=1e-6):
    # Initialize pressure correction array
    p_prime = np.zeros_like(u_r)
    
    # Get grid dimensions
    nr, nz = cylinder.num_r, cylinder.num_z
    dr, dz = cylinder.dr, cylinder.dz
    rho = fluid.rho
    
    # Compute coefficients for Jacobi iteration
    alpha_r = 1 / (dr**2)
    alpha_z = 1 / (dz**2)
    beta = rho / dt
    
    # Initialize variables for convergence check
    iter_count = 0
    residual = tol + 1  # Start with residual larger than tolerance
    
    # Jacobi iteration loop
    while iter_count < max_iter and residual > tol:
        p_prime_old = p_prime.copy()  # Store previous iteration
        
        # Update pressure correction using Jacobi method
        for i in range(1, nr - 1):
            for j in range(1, nz - 1):
                term_r = alpha_r * (p_prime_old[i+1, j] + p_prime_old[i-1, j])
                term_z = alpha_z * (p_prime_old[i, j+1] + p_prime_old[i, j-1])
                source_term = beta * ((u_r[i, j] - u_r[i-1, j]) / dr + (u_z[i, j] - u_z[i, j-1]) / dz)
                
                p_prime[i, j] = 0.5 * (term_r + term_z - source_term)
        
        # Compute residual for convergence check
        residual = np.linalg.norm(p_prime - p_prime_old) / np.sqrt(nr * nz)
        
        # Increment iteration count
        iter_count += 1
    
    # Print convergence information
    if iter_count < max_iter:
        print(f"Pressure-Poisson equation converged in {iter_count} iterations with residual {residual:.6e}")
    else:
        print(f"Pressure-Poisson equation did not converge after {max_iter} iterations, residual is {residual:.6e}")
    
    return p_prime

def starred_velocities(cylinder, fluid, dt):
    u_r, u_z, u_theta, nu, dr, dtheta, dz = cylinder.u, cylinder.w, cylinder.v, fluid.nu, cylinder.dr, cylinder.dtheta, cylinder.dz
    radius = cylinder.radial_points
    # Ensure radius has no zero values to avoid division by zero
    radius[radius == 0] = np.finfo(float).eps

    # Create empty arrays for the tentative velocities
    u_r_star = np.zeros_like(u_r)
    u_z_star = np.zeros_like(u_z)
    
    # Compute the first and second derivatives
    d1_r_u_r, d1_z_u_r = d1(u_r, dr, dtheta, dz)
    d1_r_u_z, d1_z_u_z = d1(u_z, dr, dtheta, dz)
    d2_r_u_r, d2_z_u_r = d2(u_r, dr, dtheta, dz)
    d2_r_u_z, d2_z_u_z = d2(u_z, dr, dtheta, dz)
    
    # Loop over the grid points to compute the tentative velocities (excluding boundaries)
    for i in range(1, u_r.shape[0] - 1):
        for j in range(1, u_r.shape[1] - 1):  # theta dimension (excluding boundaries)
            for k in range(1, u_r.shape[2] - 1):
                # Radial velocity
                convective_term_r = (
                    -u_r[i, j, k] * d1_r_u_r[i, j, k] 
                    - u_z[i, j, k] * d1_z_u_r[i, j, k] 
                    + (u_theta[i, j, k] ** 2) / radius[i]
                )
                diffusive_term_r = nu * (
                    d2_r_u_r[i, j, k] 
                    + (1 / radius[i]) * d1_r_u_r[i, j, k] 
                    - (u_r[i, j, k] / radius[i] ** 2) 
                    + d2_z_u_r[i, j, k]
                )
                u_r_star[i, j, k] = u_r[i, j, k] + dt * (convective_term_r + diffusive_term_r)
                
                # Axial velocity
                convective_term_z = (
                    -u_r[i, j, k] * d1_r_u_z[i, j, k] 
                    - u_z[i, j, k] * d1_z_u_z[i, j, k]
                )
                diffusive_term_z = nu * (
                    d2_r_u_z[i, j, k] 
                    + (1 / radius[i]) * d1_r_u_z[i, j, k] 
                    + d2_z_u_z[i, j, k]
                )
                u_z_star[i, j, k] = u_z[i, j, k] + dt * (convective_term_z + diffusive_term_z)
    return u_r_star, u_z_star

def laplacian_cylindrical(field, dr, dtheta, dz):
    laplacian_r = ((np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0))) / (dr**2)
    laplacian_theta = ((1 / field.shape[1]**2) * (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1))) / (dtheta**2)
    laplacian_z = ((np.roll(field, -1, axis=2) - 2 * field + np.roll(field, 1, axis=2))) / (dz**2)
    laplacian_field = laplacian_r + laplacian_theta + laplacian_z
    return laplacian_field

def d1(field, dr, dtheta, dz): # first derivative
    d1_r = ((np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0))) / (dr*2)
    d1_z = ((np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2))) / (dz*2)
    return d1_r, d1_z

def d2(field, dr, dtheta, dz): # second derivative
    d2_r = ((np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0))) / (dr**2)
    d2_z = ((np.roll(field, -1, axis=2) - 2 * field + np.roll(field, 1, axis=2))) / (dz**2)
    return d2_r, d2_z

def makeResultDirectories(wipe=False):
    cwdir = os.getcwd()
    dir_path = os.path.join(cwdir, "VTK_Results")
    
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    else:
        if wipe:
            os.chdir(dir_path)
            filelist = os.listdir()
            for file in filelist:
                os.remove(file)
    
    os.chdir(cwdir)
    cwdir = os.getcwd()
    dir_path = os.path.join(cwdir, "Results")
    
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    else:
        if wipe:
            os.chdir(dir_path)
            filelist = os.listdir()
            for file in filelist:
                os.remove(file)
    
    os.chdir(cwdir)

def writeToVTK(cylinder, i, interval):
    if i % interval == 0:
        vtk_dir_path = os.path.join(os.getcwd(), "VTK_Results")
        txt_dir_path = os.path.join(os.getcwd(), "Results")
        os.makedirs(vtk_dir_path, exist_ok=True)
        os.makedirs(txt_dir_path, exist_ok=True)

        vtk_filename = f"FluidSimulation_{i}.vtk"
        vtk_path = os.path.join(vtk_dir_path, vtk_filename)

        txt_u_r_filename = f"u_r_{i}.txt"
        txt_u_theta_filename = f"u_theta_{i}.txt"
        txt_u_z_filename = f"u_z_{i}.txt"
        txt_p_filename = f"p_{i}.txt"
        txt_u_r_path = os.path.join(txt_dir_path, txt_u_r_filename)
        txt_u_theta_path = os.path.join(txt_dir_path, txt_u_theta_filename)
        txt_u_z_path = os.path.join(txt_dir_path, txt_u_z_filename)
        txt_p_path = os.path.join(txt_dir_path, txt_p_filename)

        # Extract the required fields from the cylinder object
        u_r = cylinder.u
        u_theta = cylinder.v
        u_z = cylinder.w
        p = cylinder.p

        # Replace NaN values with 0
        u_r[np.isnan(u_r)] = 0
        u_theta[np.isnan(u_theta)] = 0
        u_z[np.isnan(u_z)] = 0
        p[np.isnan(p)] = 0

        # Save u_r, u_theta, u_z, p as text files
        np.savetxt(txt_u_r_path, u_r.flatten())
        np.savetxt(txt_u_theta_path, u_theta.flatten())
        np.savetxt(txt_u_z_path, u_z.flatten())
        np.savetxt(txt_p_path, p.flatten())

        # Get grid dimensions
        nr, ntheta, nz = cylinder.num_r, cylinder.num_theta, cylinder.num_z

        # Create a structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nr, nz, ntheta)

        # Create VTK arrays for velocity components
        u_r_array = vtk.vtkDoubleArray()
        u_r_array.SetName("u_r")
        u_z_array = vtk.vtkDoubleArray()
        u_z_array.SetName("u_z")
        u_theta_array = vtk.vtkDoubleArray()
        u_theta_array.SetName("u_theta")
        
        # Flatten the arrays (VTK requires flat arrays)
        u_r_flat = u_r.flatten(order='F')  # Fortran order to match VTK's column-major order
        u_z_flat = u_z.flatten(order='F')
        u_theta_flat = u_theta.flatten(order='F')

        # Set data to VTK arrays
        u_r_array.SetArray(u_r_flat, len(u_r_flat), 1)
        u_z_array.SetArray(u_z_flat, len(u_z_flat), 1)
        u_theta_array.SetArray(u_theta_flat, len(u_theta_flat), 1)
        
        # Add arrays to grid
        grid.GetPointData().AddArray(u_r_array)
        grid.GetPointData().AddArray(u_z_array)
        grid.GetPointData().AddArray(u_theta_array)
        
        # Write to VTK file
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(vtk_path)
        writer.SetInputData(grid)
        writer.Write()
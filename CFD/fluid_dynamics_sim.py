"""
Written by Jordan Low Jun Yi; NUS BME Y0
Created on 6/28/2024
Hours spent: 12

Read README for usage details
"""

import fluid_dynamics as fs

def main():
    # Set cylinder, fluid, CFL, iterations and VTK export interval
    cylinder = fs.Cylinder(radius=1.0, length=10.0, num_r=50, num_theta=50, num_z=50)
    fluid = fs.Fluid(rho=1.225, nu=1.81e-5)  # Air at sea level
    CFL = 0.5
    iterations = 100
    VTK_interval = 10
    
    # Set boundary conditions
    vInflow, vOutflow = 1, 1
    pInflow = 1

    # Do not edit past this
    fs.makeResultDirectories(True)
    dt = fs.timeStep(CFL, cylinder)

    for i in range(iterations):
        cylinder.setVelocityBoundaries(vInflow, vOutflow)
        cylinder.setPressureBoundaries(pInflow) 
        u_r_star, u_z_star = fs.starred_velocities(cylinder, fluid, dt)
        p_prime = fs.solve_pressure_poisson(u_r_star, u_z_star, cylinder, fluid, dt)
        cylinder.u += dt * (-(1 / fluid.rho) * fs.d1(p_prime, cylinder.dr, cylinder.dtheta, cylinder.dz)[0])
        cylinder.w += + dt * (-(1 / fluid.rho) * fs.d1(p_prime, cylinder.dr, cylinder.dtheta, cylinder.dz)[1])
        # if i forget, u is velocity_r, v is velocity_theta, w is velocity_z
        fs.writeToVTK(cylinder, i, VTK_interval)

    print("Simulation complete.")

if __name__ == "__main__":
    main()

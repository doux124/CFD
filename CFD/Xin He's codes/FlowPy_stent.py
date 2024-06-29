#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:57:17 2024

@author: yangxinhe
"""


import numpy as np
import os

class Space:
    def __init__(self):
        pass
    
    def SetSourceTerm(self,S_x=0,S_y=0):
        self.S_x=S_x
        self.S_y=S_y
    
    def CreateMesh(self,rowpts,colpts):
        #Domain gridpoints
        self.rowpts = rowpts
        self.colpts = colpts
        
        #Velocity matrices
        self.u = np.zeros((self.rowpts+2,self.colpts+2))
        self.v = np.zeros((self.rowpts+2,self.colpts+2))
        self.u_star = np.zeros((self.rowpts+2,self.colpts+2))
        self.v_star = np.zeros((self.rowpts+2,self.colpts+2))
        self.u_next = np.zeros((self.rowpts+2,self.colpts+2))
        self.v_next = np.zeros((self.rowpts+2,self.colpts+2))
        self.u_c = np.zeros((self.rowpts,self.colpts))
        self.v_c = np.zeros((self.rowpts,self.colpts))
        self.velocity_c = np.zeros((self.rowpts,self.colpts))
        
        #Pressure matrices
        self.p = np.zeros((self.rowpts+2,self.colpts+2))
        self.p_c = np.zeros((self.rowpts,self.colpts))
        
        #Set default source term
        self.SetSourceTerm() 
        
    def SetDeltas(self,breadth,length):
        self.dx = length/(self.colpts - 1)
        self.dy = breadth/(self.rowpts - 1)
        
    def SetInitialU(self,U):
        self.u = U * self.u
        
    def SetInitialV(self,V):
        self.v = V * self.v
        
    def SetInitialP(self,P):
        self.p = P * self.p
        

class Fluid: 
    def __init__(self,rho,mu):
        self.SetFluidProperties(rho,mu)
    
    def SetFluidProperties(self,rho,mu): # a bit redundant?
        self.rho = rho # density of fluid
        self.mu = mu # viscosity of fluid        
    
def SetUBoundary(space,top_value,bottom_value,left_value,vert_value,void_value):
    space.u[:,0] = left_value
    # set values for the "pushed outwards" parts
    space.u[0,0:int(1/5*(space.colpts+2))] = top_value
    space.u[0,int(2/5*(space.colpts+2)):int(3/5*(space.colpts+2))] = top_value
    space.u[0,int(4/5*(space.colpts+2)):] = top_value
    
    # set values for the "pushed inwards" parts
    space.u[int(1/10*(space.rowpts+2)),int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = top_value
    space.u[int(1/10*(space.colpts+2)),int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = top_value
    
    # set values for the "vertical" walls
    space.u[0:int(1/10*(space.rowpts+2)),int(1/5*(space.colpts+2))] = vert_value
    space.u[0:int(1/10*(space.rowpts+2)),int(2/5*(space.colpts+2))] = vert_value
    space.u[0:int(1/10*(space.rowpts+2)),int(3/5*(space.colpts+2))] = vert_value
    space.u[0:int(1/10*(space.rowpts+2)),int(4/5*(space.colpts+2))] = vert_value
    
    # do the same for the bottom parts
    space.u[-1,0:int(1/5*(space.colpts+2))] = bottom_value
    space.u[-1,int(2/5*(space.colpts+2)):int(3/5*(space.colpts+2))] = bottom_value
    space.u[-1,int(4/5*(space.colpts+2)):] = bottom_value
    
    space.u[int(space.rowpts+2-1),int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = bottom_value
    space.u[int(space.colpts+2-1),int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = bottom_value
    
    space.u[int(9/10*(space.rowpts+2)):,int(1/5*(space.colpts+2))] = vert_value
    space.u[int(9/10*(space.rowpts+2)):,int(2/5*(space.colpts+2))] = vert_value
    space.u[int(9/10*(space.rowpts+2)):,int(3/5*(space.colpts+2))] = vert_value
    space.u[int(9/10*(space.rowpts+2)):,int(4/5*(space.colpts+2))] = vert_value
    
    # void spaces
    space.u[0:int(1/10*(space.rowpts+2)),int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = void_value
    space.u[0:int(1/10*(space.rowpts+2)),int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = void_value
    space.u[int(9/10*(space.rowpts+2)):,int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = void_value
    space.u[int(9/10*(space.rowpts+2)):,int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = void_value    
    
def SetVBoundary(space,top_value,bottom_value,left_value,vert_value,void_value):
    space.v[:,0] = left_value
    # set values for the "pushed outwards" parts
    space.v[0,0:int(1/5*(space.colpts+2))] = top_value
    space.v[0,int(2/5*(space.colpts+2)):int(3/5*(space.colpts+2))] = top_value
    space.v[0,int(4/5*(space.colpts+2)):] = top_value
    
    # set values for the "pushed inwards" parts
    space.v[int(1/10*(space.rowpts+2)),int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = top_value
    space.v[int(1/10*(space.colpts+2)),int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = top_value
    
    # set values for the "vertical" walls
    space.v[0:int(1/10*(space.rowpts+2)),int(1/5*(space.colpts+2))] = vert_value
    space.v[0:int(1/10*(space.rowpts+2)),int(2/5*(space.colpts+2))] = vert_value
    space.v[0:int(1/10*(space.rowpts+2)),int(3/5*(space.colpts+2))] = vert_value
    space.v[0:int(1/10*(space.rowpts+2)),int(4/5*(space.colpts+2))] = vert_value
    
    # do the same for the bottom parts
    space.v[-1,0:int(1/5*(space.colpts+2))] = bottom_value
    space.v[-1,int(2/5*(space.colpts+2)):int(3/5*(space.colpts+2))] = bottom_value
    space.v[-1,int(4/5*(space.colpts+2)):] = bottom_value
    
    space.v[int(space.rowpts+2-1),int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = bottom_value
    space.v[int(space.colpts+2-1),int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = bottom_value
    
    space.v[int(9/10*(space.rowpts+2)):,int(1/5*(space.colpts+2))] = vert_value
    space.v[int(9/10*(space.rowpts+2)):,int(2/5*(space.colpts+2))] = vert_value
    space.v[int(9/10*(space.rowpts+2)):,int(3/5*(space.colpts+2))] = vert_value
    space.v[int(9/10*(space.rowpts+2)):,int(4/5*(space.colpts+2))] = vert_value
    
    # void spaces
    space.v[0:int(1/10*(space.rowpts+2)),int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = void_value
    space.v[0:int(1/10*(space.rowpts+2)),int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = void_value
    space.v[int(9/10*(space.rowpts+2)):,int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = void_value
    space.v[int(9/10*(space.rowpts+2)):,int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = void_value    
    
def SetPBoundary(space,left_value,right_value,void_value):
    space.p[:,0] = left_value
    space.p[:,-1] = right_value
    # void spaces
    space.p[0:int(1/10*(space.rowpts+2)),int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = void_value
    space.p[0:int(1/10*(space.rowpts+2)),int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = void_value
    space.p[int(9/10*(space.rowpts+2)):,int(1/5*(space.colpts+2)):int(2/5*(space.colpts+2))] = void_value
    space.p[int(9/10*(space.rowpts+2)):,int(3/5*(space.colpts+2)):int(4/5*(space.colpts+2))] = void_value  

    
def SetTimeStep(CFL,space,fluid): # Do we need fluid here?
    with np.errstate(divide='ignore'): # ignore the zero division error first
        dt = CFL/np.sum([np.max(space.u)/space.dx,\
                           np.max(space.v)/space.dy])
    #Escape condition if dt is infinity due to zero velocity initially
    if np.isinf(dt):
        dt = CFL*(space.dx+space.dy)
    space.dt=dt
    
def GetStarredVelocities(space,fluid):
    rows = int(space.rowpts) # Number of rows and columns can only be in integers
    cols = int(space.colpts)
    u = space.u.astype(float,copy=False)
    v = space.v.astype(float,copy=False)
    dx = float(space.dx)
    dy = float(space.dy)
    dt = float(space.dt)
    S_x = float(space.S_x) # acceleration from external force acting on the fluid
    S_y = float(space.S_y)
    rho = float(fluid.rho)
    mu = float(fluid.mu)
    
    # Copy u and v to new variables u_star and v_star
    u_star = u.copy()
    v_star = v.copy()
    
    # Calculate derivatives of u and v using the Taylor series expansion. 
    # Numpy vectorization saves us from using slower 'for loops' to go 
    # over each element in the u and v matrices
    u1_y = (u[2:rows+2,1:cols+1] - u[0:rows,1:cols+1])/(2*dy)
    u1_x = (u[1:rows+1,2:cols+2] - u[1:rows+1,0:cols])/(2*dx)
    u2_y = (u[2:rows+2,1:cols+1] - 2*u[1:rows+1,1:cols+1] + u[0:rows,1:cols+1])/(dy**2)
    u2_x = (u[1:rows+1,2:cols+2] - 2*u[1:rows+1,1:cols+1] + u[1:rows+1,0:cols])/(dx**2)
    v_face = (v[1:rows+1,1:cols+1] + v[1:rows+1,0:cols] + v[2:,1:cols+1] + v[2:,0:cols])/4
    u_star[1:rows+1,1:cols+1] = u[1:rows+1,1:cols+1]\
         - dt*(u[1:rows+1,1:cols+1]*u1_x + v_face*u1_y) + \
             (dt*(mu/rho)*(u2_x+u2_y)) + (dt*S_x) 
            
            
    v1_y = (v[2:rows+2,1:cols+1] - v[0:rows,1:cols+1])/(2*dy)
    v1_x = (v[1:rows+1,2:cols+2]-v[1:rows+1,0:cols])/(2*dx)
    v2_y = (v[2:rows+2,1:cols+1]-2*v[1:rows+1,1:cols+1]+v[0:rows,1:cols+1])/(dy**2)
    v2_x = (v[1:rows+1,2:cols+2]-2*v[1:rows+1,1:cols+1]+v[1:rows+1,0:cols])/(dx**2)
    u_face = (u[1:rows+1,1:cols+1]+u[1:rows+1,2:]+u[0:rows,1:cols+1]+u[0:rows,2:])/4
    v_star[1:rows+1,1:cols+1] = v[1:rows+1,1:cols+1] - dt*(u_face*v1_x + \
        v[1:rows+1,1:cols+1]*v1_y)+ \
         (dt*(mu/rho)*(v2_x+v2_y))+(dt*S_y)
            
    # Save the calculated starred velocities to the space object 
    space.u_star = u_star.copy()
    space.v_star = v_star.copy() 
    
# The second function is used to iteratively solve the pressure Poisson equation from the starred velocities 
# to calculate pressure at t+delta_t
def SolvePressurePoisson(space,fluid,left,right,void):
    #Save object attributes as local variable with explicity typing for improved readability
    rows = int(space.rowpts)
    cols = int(space.colpts)
    u_star = space.u_star.astype(float,copy=False)
    v_star = space.v_star.astype(float,copy=False)
    p = space.p.astype(float,copy=False)
    dx = float(space.dx)
    dy = float(space.dy)
    dt = float(space.dt)
    rho = float(fluid.rho)
    factor = 1/(2/dx**2+2/dy**2)
    
    
    # Define initial error and tolerance for convergence (error > tol necessary initially)
    error = 1
    tol = 1e-3
    
    #Evaluate derivative of starred velocities
    ustar1_x = (u_star[1:rows+1,2:cols+2] - u_star[1:rows+1,0:cols])/(2*dx)
    vstar1_y = (v_star[2:rows+2,1:cols+1] - v_star[0:rows,1:cols+1])/(2*dy)
    #print(ustar1_x)

   #Continue iterative solution until error becomes smaller than tolerance
    i = 0
    while(error > tol):
        i += 1
       
       # Save current pressure as p_old
        p_old = p.astype(float,copy=True)
       
       # Evaluate second derivative of pressure from p_old
        p2_xy = (p_old[2:rows+2,1:cols+1] + p_old[0:rows,1:cols+1])/dy**2 + \
            (p_old[1:rows+1,2:cols+2] + p_old[1:rows+1,0:cols])/dx**2
       
       # Calculate new pressure 
        p[1:rows+1,1:cols+1] = (p2_xy)*factor - (rho*factor/dt)*(ustar1_x + vstar1_y)
       
       # Find maximum error between old and new pressure matrices
        error = np.max(abs(p-p_old))
       
       # Apply pressure boundary conditions
        SetPBoundary(space,left,right,void)
       
       # Escape condition in case solution does not converge after 500 iterations
        if(i > 500):
            break
    space.p = p.copy()  
    #print(i)
        
#The third function is used to calculate the velocities at timestep t+delta_t using the pressure at t+delta_t and starred velocities
def SolveMomentumEquation(space,fluid):
    #Save object attributes as local variable with explicity typing for improved readability
    rows = int(space.rowpts)
    cols = int(space.colpts)
    u_star = space.u_star.astype(float,copy=False)
    v_star = space.v_star.astype(float,copy=False)
    p = space.p.astype(float,copy=False)
    dx = float(space.dx)
    dy = float(space.dy)
    dt = float(space.dt)
    rho = float(fluid.rho)
    u = space.u.astype(float,copy=False)
    v = space.v.astype(float,copy=False)
    
    #Evaluate first derivative of pressure in x direction
    p1_x = (p[1:rows+1,2:cols+2] - p[1:rows+1,0:cols])/(2*dx)
    #Calculate u at next timestep
    u[1:rows+1,1:cols+1] = u_star[1:rows+1,1:cols+1] - (dt/rho)*p1_x

    #Evaluate first derivative of pressure in y direction
    p1_y = (p[2:rows+2,1:cols+1] - p[0:rows,1:cols+1])/(2*dy)
    #Calculate v at next timestep
    v[1:rows+1,1:cols+1] = v_star[1:rows+1,1:cols+1] - (dt/rho)*p1_y
    
def SetCentrePUV(space):
    space.p_c = space.p[1:-1,1:-1]
    space.u_c = space.u[1:-1,1:-1]
    space.v_c = space.v[1:-1,1:-1]
    space.velocity_c = np.sqrt((space.u_c)**2 + (space.v_c)**2)
    
def MakeResultDirectory(wipe=False): # default value of wipe is False
    #Get path to the Result directory
    cwdir = os.getcwd() # get current working directory
    dir_path = os.path.join(cwdir,"Result")
    #If directory does not exist, make it
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path,exist_ok = True)
    else:
        #If wipe is True, remove files present in the directory
        if wipe:
            os.chdir(dir_path) # change directory
            filelist = os.listdir() # get the list of all files from that directory
            for file in filelist:
                os.remove(file)
    
    os.chdir(cwdir)
    
def WriteToFile(space,iteration,interval):
    if(iteration%interval==0):
        dir_path = os.path.join(os.getcwd(),"Result")
        filename = "PUV{0}.txt".format(iteration)
        path = os.path.join(dir_path,filename)
        with open(path,"w") as f:
            for i in range(space.rowpts):
                for j in range(space.colpts):
                    f.write("{}\t{}\t{}\t{}\n".format(space.p_c[i,j],space.u_c[i,j],space.v_c[i,j],space.velocity_c[i,j]))
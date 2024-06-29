#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:15:25 2024

@author: yangxinhe
"""

import numpy as np
import os

# Maybe should remove v?

# Creates a class called "Boundary"
# class Boundary:
#     def __init__(self,boundary_value):
#         self.DefineBoundary(boundary_value) 
#         # Creates a method called "DefineBoundary"
    
#     def DefineBoundary(self,boundary_value):
#         self.value = boundary_value

# creates a class called "Space"        
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
        self.u=np.zeros((self.rowpts+2,self.colpts+2))
        self.v=np.zeros((self.rowpts+2,self.colpts+2))
        self.u_star=np.zeros((self.rowpts+2,self.colpts+2))
        self.v_star=np.zeros((self.rowpts+2,self.colpts+2))
        self.u_next=np.zeros((self.rowpts+2,self.colpts+2))
        self.v_next=np.zeros((self.rowpts+2,self.colpts+2))
        self.u_c=np.zeros((self.rowpts,self.colpts))
        self.v_c=np.zeros((self.rowpts,self.colpts))
        
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
        
    def SetInitialP(self,P):
        self.p = P * self.p
        

class Fluid: 
    def __init__(self,rho,mu):
        self.SetFluidProperties(rho,mu)
    
    def SetFluidProperties(self,rho,mu): # a bit redundant?
        self.rho = rho # density of fluid
        self.mu = mu # viscosity of fluid        
    
    

def SetUBoundaryfq(space,lines_value,left_value):
    space.u[int((space.rowpts+2)*3/8),1:int((space.colpts+2)/2)] = 0
    space.u[int((space.rowpts+2)*5/8),1:int((space.colpts+2)/2)] = 0
    # upper rectangle
    space.u[0:int((space.rowpts+2)*3/8),0:int((space.colpts+2)/2)] = 0
    # lower rectangle
    space.u[int((space.rowpts+2)*5/8):int(space.rowpts+2),0:int((space.colpts+2)/2)] = 0
    space.u[int((space.rowpts+2)*3/8):int((space.rowpts+2)*5/8),0] = left_value  
   


def SetUBoundarysq(space,lines_value):
    # upper outer triangle
    part = np.rot90(space.u[0:int((space.rowpts+2)*3/8),int((space.colpts+2)/2):space.colpts+2])
    np.fill_diagonal(part,lines_value)
    dimpa = np.shape(part)
    #print(dimpa)
    part_idx = np.tril_indices(dimpa[0],-1)
    #print(part_idx)
    part[part_idx] = 0
    part1 = np.rot90(part,3)
    space.u[0:int((space.rowpts+2)*3/8),int((space.colpts+2)/2):space.colpts+2] = part1
    # upper inner triangle
    unit = np.rot90(space.u[int((space.rowpts+2)/4):int((space.rowpts+2)/2),int((space.colpts+2)*2/3):int(space.colpts+2)])
    np.fill_diagonal(unit,lines_value)
    dimu = np.shape(unit)
    unit_idx = np.triu_indices(dimu[0],1)
    unit[unit_idx] = 0
    unit1 = np.rot90(unit,3) 
    space.u[int((space.rowpts+2)/4):int((space.rowpts+2)/2),int((space.colpts+2)*2/3):int(space.colpts+2)] = unit1

    
    
def SetUBoundarytq(space,lines_value):
    # lower outer triangle
    piece = space.u[int((space.rowpts+2)*5/8):space.rowpts+2,int((space.colpts+2)/2):space.colpts+2]
    np.fill_diagonal(piece,lines_value)
    dimp = np.shape(piece)
    #print(dimp)
    piece_idx = np.tril_indices(dimp[0],-1)
    piece[piece_idx] = 0
    space.u[int((space.rowpts+2)*5/8):space.rowpts+2,int((space.colpts+2)/2):space.colpts+2] = piece
    # lower inner triangle
    chunk = space.u[int((space.rowpts+2)/2):int((space.rowpts+2)*3/4),int((space.colpts+2)*2/3):space.colpts+2]
    np.fill_diagonal(chunk,lines_value)
    dimc = np.shape(chunk)
    chunk_idx = np.triu_indices(dimc[0],1)
    chunk[chunk_idx] = 0
    space.u[int((space.rowpts+2)/2):int((space.rowpts+2)*3/4),int((space.colpts+2)*2/3):space.colpts+2] = chunk   

    
    

def SetVBoundaryfq(space,lines_value,left_value):
    space.v[int((space.rowpts+2)*3/8),1:int((space.colpts+2)/2)] = 0
    space.v[int((space.rowpts+2)*5/8),1:int((space.colpts+2)/2)] = 0
    # upper rectangle
    space.v[0:int((space.rowpts+2)*3/8),0:int((space.colpts+2)/2)] = 0
    # lower rectangle
    space.v[int((space.rowpts+2)*5/8):int(space.rowpts+2),0:int((space.colpts+2)/2)] = 0
    space.v[int((space.rowpts+2)*3/8):int((space.rowpts+2)*5/8),0] = left_value 
         
    
def SetVBoundarysq(space,lines_value):
    # upper outer triangle
    part = np.rot90(space.v[0:int((space.rowpts+2)*3/8),int((space.colpts+2)/2):space.colpts+2])
    np.fill_diagonal(part,lines_value)
    dimpa = np.shape(part)
    #print(dimpa)
    part_idx = np.tril_indices(dimpa[0],-1)
    #print(part_idx)
    part[part_idx] = 0
    part1 = np.rot90(part,3)
    space.v[0:int((space.rowpts+2)*3/8),int((space.colpts+2)/2):space.colpts+2] = part1
    # upper inner triangle
    unit = np.rot90(space.v[int((space.rowpts+2)/4):int((space.rowpts+2)/2),int((space.colpts+2)*2/3):int(space.colpts+2)])
    np.fill_diagonal(unit,lines_value)
    dimu = np.shape(unit)
    unit_idx = np.triu_indices(dimu[0],1)
    unit[unit_idx] = 0
    unit1 = np.rot90(unit,3) 
    space.v[int((space.rowpts+2)/4):int((space.rowpts+2)/2),int((space.colpts+2)*2/3):int(space.colpts+2)] = unit1


    

def SetVBoundarytq(space,lines_value):
    # lower outer triangle
    piece = space.v[int((space.rowpts+2)*5/8):space.rowpts+2,int((space.colpts+2)/2):space.colpts+2]
    np.fill_diagonal(piece,lines_value)
    dimp = np.shape(piece)
    #print(dimp)
    piece_idx = np.tril_indices(dimp[0],-1)
    piece[piece_idx] = 0
    space.v[int((space.rowpts+2)*5/8):space.rowpts+2,int((space.colpts+2)/2):space.colpts+2] = piece
    # lower inner triangle
    chunk = space.v[int((space.rowpts+2)/2):int((space.rowpts+2)*3/4),int((space.colpts+2)*2/3):space.colpts+2]
    np.fill_diagonal(chunk,lines_value)
    dimc = np.shape(chunk)
    chunk_idx = np.triu_indices(dimc[0],1)
    chunk[chunk_idx] = 0
    space.v[int((space.rowpts+2)/2):int((space.rowpts+2)*3/4),int((space.colpts+2)*2/3):space.colpts+2] = chunk  

       
    
def SetPBoundaryfq(space,left_value,right_value1,right_value2):
    space.p[int((space.rowpts+2)*(3/8)):int((space.rowpts+2)*(5/8)),0] = left_value
    space.p[0:int((space.rowpts+2)/4),-1] = right_value1
    space.p[int((space.rowpts+2)*(3/4)):space.colpts+2,-1] = right_value2
    # upper rectangle
    space.p[0:int((space.rowpts+2)*3/8),1:int((space.colpts+2)/2)] = 0
    # lower rectangle
    space.p[int((space.rowpts+2)*5/8):int(space.rowpts+2),0:int((space.colpts+2)/2)] = 0 

    
def SetPBoundarysq(space):   
    # upper outer triangle
    part = np.rot90(space.p[0:int((space.rowpts+2)*3/8),int((space.colpts+2)/2):space.colpts+2])
    #np.fill_diagonal(part,lines_value)
    dimpa = np.shape(part)
    #print(dimpa)
    part_idx = np.tril_indices(dimpa[0],-1)
    #print(part_idx)
    part[part_idx] = 0
    part1 = np.rot90(part,3)
    space.p[0:int((space.rowpts+2)*3/8),int((space.colpts+2)/2):space.colpts+2] = part1
    
    # upper inner triangle
    unit = np.rot90(space.p[int((space.rowpts+2)/4):int((space.rowpts+2)/2),int((space.colpts+2)*2/3):space.colpts+2])
    #np.fill_diagonal(unit,lines_value)
    dimu = np.shape(unit)
    unit_idx = np.triu_indices(dimu[0],1)
    unit[unit_idx] = 0
    unit1 = np.rot90(unit,3) 
    space.p[int((space.rowpts+2)/4):int((space.rowpts+2)/2),int((space.colpts+2)*2/3):space.colpts+2] = unit1
 
    
    
def SetPBoundarytq(space):
    # lower outer triangle
    piece = space.p[int((space.rowpts+2)*5/8):space.rowpts+2,int((space.colpts+2)/2):space.colpts+2]
    #np.fill_diagonal(piece,lines_value)
    dimp = np.shape(piece)
    #print(dimp)
    piece_idx = np.tril_indices(dimp[0],-1)
    piece[piece_idx] = 0
    space.p[int((space.rowpts+2)*5/8):space.rowpts+2,int((space.colpts+2)/2):space.colpts+2] = piece 

    # lower inner triangle
    chunk = space.p[int((space.rowpts+2)/2):int((space.rowpts+2)*3/4),int((space.colpts+2)*2/3):space.colpts+2]
    #np.fill_diagonal(chunk,lines_value)
    dimc = np.shape(chunk)
    chunk_idx = np.triu_indices(dimc[0],1)
    chunk[chunk_idx] = 0
    space.p[int((space.rowpts+2)/2):int((space.rowpts+2)*3/4),int((space.colpts+2)*2/3):space.colpts+2] = chunk
    
    
def SetTimeStep(CFL,space,fluid):
    with np.errstate(divide='ignore'):
        dt=CFL/np.sum([np.max(space.u)/space.dx,\
                           np.max(space.v)/space.dy])
    #Escape condition if dt is infinity due to zero velocity initially
    if np.isinf(dt):
        dt=CFL*(space.dx+space.dy)
    space.dt=dt
    
def GetStarredVelocities(space,fluid):
    rows = int(space.rowpts) # Number of rows and columns can only be in integers
    cols = int(space.colpts)
    u = space.u.astype(float,copy=False)
    v = space.v.astype(float,copy=False)
    dx = float(space.dx)
    dy = float(space.dy)
    dt = float(space.dt)
    S_x = float(space.S_x) # external force acting on the fluid
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
def SolvePressurePoisson(space,fluid,left,right1,right2):
    #Save object attributes as local variable with explicit typing for improved readability
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
    while error > tol:
        i += 1
       
       # Save current pressure as p_old
        p_old = p.astype(float,copy=True)
       
       # Evaluate second derivative of pressure from p_old
        p2_xy = (p_old[2:rows+2,1:cols+1] + p_old[0:rows,1:cols+1])/dy**2 + \
            (p_old[1:rows+1,2:cols+2] + p_old[1:rows+1,0:cols])/dx**2
       
       # Calculate new pressure 
        p[1:rows+1,1:cols+1]=(p2_xy)*factor-(rho*factor/dt)*(ustar1_x+vstar1_y)
       
       # Find maximum error between old and new pressure matrices
        error = np.max(abs(p-p_old))
       
       # Apply pressure boundary conditions
        SetPBoundaryfq(space,left,right1,right2)
        SetPBoundarysq(space)
        SetPBoundarytq(space)
       
       # Escape condition in case solution does not converge after 500 iterations
        if i > 500:
            break
    space.p = p.copy() 
    
    
        
    
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
                    f.write("{}\t{}\t{}\n".format(space.p_c[i,j],space.u_c[i,j],space.v_c[i,j]))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 01:09:26 2024

@author: yangxinhe
"""

import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#### Simulation inputs
rowpts = 240
colpts = 180
length = 3
breadth = 4

# Go to the Result directory
cwdir = os.getcwd()
dir_path = os.path.join(cwdir,"Result")
os.chdir(dir_path)

# Go through files in the directory and store filenames
filenames = []
iterations = []
for root,dirs,files in os.walk(dir_path):
    for datafile in files:
        if "PUV" in datafile:
            filenames.append(datafile)
            # print(filenames)
            no_ext_file=datafile.replace(".txt","").strip()
            iter_no=int(no_ext_file.split("V")[-1])
            iterations.append(iter_no)
            
# Discern the final iteration and interval
initial_iter = np.min(iterations)            
final_iter = np.max(iterations)
inter = (final_iter - initial_iter)/(len(iterations)-1)
number_of_frames = len(iterations)
sorted_iterations = np.sort(iterations)

def read_datafile(iteration):
    #Set filename and path according to given iteration
    filename="PUV{0}.txt".format(iteration)
    filepath=os.path.join(dir_path,filename)
    #Load text file as numpy array
    arr=np.loadtxt(filepath,delimiter="\t")
    rows,cols=arr.shape
    #Define empty arrays for pressure and velocities
    p_p=np.zeros((rowpts,colpts))
    u_p=np.zeros((rowpts,colpts))
    v_p=np.zeros((rowpts,colpts))
    #Organize imported array into variables
    p_arr=arr[:,0]
    u_arr=arr[:,1]
    v_arr=arr[:,2]
    
    #Reshape 1D data into 2D
    p_p=p_arr.reshape((rowpts,colpts))
    u_p=u_arr.reshape((rowpts,colpts))
    v_p=v_arr.reshape((rowpts,colpts))
    
    return p_p,u_p,v_p

#Create mesh for X and Y inputs to the figure
x=np.linspace(0,length,colpts)
y=np.linspace(0,breadth,rowpts)
[X,Y]=np.meshgrid(x,y)

#Determine indexing for stream plot (10 points only)
index_cut_x=int(colpts/15)
index_cut_y=int(rowpts/30)

#Create blank figure
fig=plt.figure(figsize=(17,16))
ax=plt.axes(xlim=(0,length),ylim=(0,breadth))

#Create initial contour and stream plot as well as color bar
p_p,u_p,v_p=read_datafile(0)
ax.set_xlim([0,length])
ax.set_ylim([0,breadth])
ax.set_xlabel("$x$",fontsize=12)
ax.set_ylabel("$y$",fontsize=12)
ax.set_title("Frame No: 0")

df1 = pd.DataFrame({'x':[0, 1.5, 3],
                   'y':[1.5, 1.5, 0]})

x1 = df1['x']
y1 = df1['y']

ax.set_xlim([0,3])
ax.set_ylim([0,4])
ax.plot(x1, y1, color = 'black')

df2 = pd.DataFrame({'x':[0, 1.5, 3],
                    'y':[2.5, 2.5, 4]})

x2 = df2['x']
y2 = df2['y']

ax.plot(x2, y2, color = 'black')

df3 = pd.DataFrame({'x':[3, 2, 3],
                    'y':[3, 2, 1]})

x3 = df3['x']
y3 = df3['y']
ax.plot(x3, y3, color = 'black')

cont=ax.contourf(X,Y,p_p)
stream=ax.quiver(X[::index_cut_y,::index_cut_x],Y[::index_cut_y,::index_cut_x],u_p[::index_cut_y,::index_cut_x],v_p[::index_cut_y,::index_cut_x],color="k")
fig.colorbar(cont,label="Pressure ($\mathregular{Nm^{-2}}$)")
fig.tight_layout()

def animate(i):
    #Print frames left to be added to the animation
    sys.stdout.write("\rFrames remaining: {0:03d}".format(len(sorted_iterations)-i))
    sys.stdout.flush()
    #Get iterations in a sequential manner through sorted_iterations
    iteration=sorted_iterations[i]
    #Use the read_datafile function to get pressure and velocities
    p_p,u_p,v_p=read_datafile(iteration)
    #Clear previous plot and make contour and stream plots for current iteration
    ax.clear()
    ax.set_xlim([0,length])
    ax.set_ylim([0,breadth])
    ax.set_xlabel("$x$",fontsize=12)
    ax.set_ylabel("$y$",fontsize=12)
    ax.set_title("Frame No: {0}".format(i))
    df1 = pd.DataFrame({'x':[0, 1.5, 3],
                       'y':[1.5, 1.5, 0]})

    x1 = df1['x']
    y1 = df1['y']

    ax.set_xlim([0,3])
    ax.set_ylim([0,4])
    ax.plot(x1, y1, color = 'black')

    df2 = pd.DataFrame({'x':[0, 1.5, 3],
                        'y':[2.5, 2.5, 4]})

    x2 = df2['x']
    y2 = df2['y']

    ax.plot(x2, y2, color = 'black')

    df3 = pd.DataFrame({'x':[3, 2, 3],
                        'y':[3, 2, 1]})

    x3 = df3['x']
    y3 = df3['y']
    ax.plot(x3, y3, color = 'black')
    
    cont=ax.contourf(X,Y,p_p)
    stream=ax.quiver(X[::index_cut_y,::index_cut_x],\
                         Y[::index_cut_y,::index_cut_x],\
                         u_p[::index_cut_y,::index_cut_x],\
                         v_p[::index_cut_y,::index_cut_x],\
                         color="k")
    return cont,stream

print("######## Making FlowPy Animation ########")
print("#########################################")
anim=animation.FuncAnimation(fig,animate,frames=number_of_frames,interval=50,blit=False)
movie_path=os.path.join(dir_path,"FluidFlowAnimation.mp4")
anim.save(r"{0}".format(movie_path))
print("\nAnimation saved as FluidFlowAnimation.mp4 in Result")
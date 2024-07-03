#!/usr/bin/env python
#Cell Swim Simulation Code
#Version 0.1.1: Point Particles with Brownian Motion and Lennard-Jones (Vectorized calculation)

#imports
import numpy as np
import pandas as pd
from sys import argv

outfile = argv[1]


#Functions
#For writing n and timestamp
def write_n_time(outfile, n, time):
    with open(outfile, 'a') as f:
        f.write(f"{n}\n")
        f.write(f"time,{time}\n")

#For appending XYZ snapshot dataframe 
def snap_add(outfile, df):
    df.to_csv(outfile, mode='a', sep=' ', header=False, index=False)

#Generate n random points within sphere of radius L for initializing positions
def gen_points(n,L):
    theta = np.random.uniform(0,2 * np.pi, n) #azimuthal angle
    phi = np.arccos(2*np.random.uniform(0,1,n)-1) #polar angle
    #spherical to cartesian
    x = L * np.sin(phi) * np.cos(theta)
    y = L * np.sin(phi) * np.sin(theta)
    z = L * np.cos(phi)
    #stack XYZ
    heads = np.stack((x,y,z), axis = -1)
    return heads

#Random force for stochastic noise
def sto(n):
    brown = co * np.random.normal(size=(n,3))
    return brown
#Vectorized LJ calculation
#takes an (n,3) matrix
def lj_vectorized(positions):
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :] #(n,n,3) vector difference rj(xyz)-ri(xyz)
    rsq = np.sum(diff**2, axis=-1)
    #fills diagonals with infinity so that the LJ due to self interaction is zero. also avoids dividing by zero
    np.fill_diagonal(rsq, np.inf) 
    six = (sigma**2/rsq)**3
    twelve = six**2
    fmag = 48 * eps * (twelve - six/2)/rsq
    forces = np.sum(fmag[:,:,np.newaxis]*diff, axis=1)
    return forces

#Projects 3D vector to 1D axis formed by point 1 and point 2
#def to_1d(vector, r1, r2):
#    #vector: 3d vector to reduce 
#    axis = r2 - r1 #vector connecting two points
#    unit_vector = axis/np.linalg.norm(axis)
#    proj = np.dot(vector, unit_vector)
#    return(proj)

#Run Parameters
n = 30 #number of cells
maxtp = 100000 #timesteps
dt = 0.01 #size of time step in s
#total run time = dt * maxtp
mass = 1.
aa = 2.0
b = 1.1 #head-tail distance
#Spherical Simulation Space
L = 30.0 #radius
dL = 0.01*L #thickness (distance from edge at which particles bounce off)
#awall = 1/((L+dL)**4 - L**4) #wall bouncing force

eta = 1.0 #viscosity
pi = np.pi #pi
gamma = 6.0*pi*eta*aa
beta = 1.0 #temperature contral 1/kBT
#constant to multiply random number for stochastic force
co = (2.0/beta/gamma*dt)**0.5

#cutoff = 10 #determine if potential is turned on or off

#Lennard-Jones Parameters
eps = 1
rstar = 1.12245295
sigma = 1 

#Clumping Force Parameters
cutoff = 1.5
rm = 8.0 #peak of potential well
del_r = 10.0 #width of potential well
r0 = 3.0
r1 = 13.0
u = 10.0 #well depth
c1 = 8 * u / del_r**2
c2 = 16 * u / del_r**4

#Active Force Paramters
t_on = 0.5 #on duration in s
t_off = 0.5 #off duration in s
switching_frequency = 2/(t_on + t_off)
vm = 5. #speed when motor is on in um/s

counter = np.zeros(n) #counts number of time steps
on_state = np.zeros(n) #stores the on or off state of motor

#Initializing positions
heads = gen_points(n,L-dL)

#Initializing velocities to 0
v = np.zeros((n,3))
time=0
with open(outfile, 'w') as f:
    f.write(f"{n}\n")
    f.write(f"time,{time}\n")
#write_n_time(outfile, n, 0)

#initial dataframe
cols = ['atomtype','x','y','z']
df = pd.DataFrame(columns=cols)
for i in heads:
    df.loc[len(df)]=['Ar',i[0],i[1],i[2]]

snap_add(outfile,df)

for step in range(maxtp):
    Fnet = np.zeros((n,3)) #reset Force to 0
    t = (step+1)*dt
    #writing XYZ file headers
    write_n_time(outfile,n,t)
    df = pd.DataFrame(columns=cols)
    LJ = lj_vectorized(heads)
    brown = sto(n)
    Fnet = LJ+brown
    #a = Fnet/mass #since mass=1, this is not necessary
    dv = Fnet*dt
    v = v+dv
    dr = v*dt
    newpos = heads+dr
    outside = np.linalg.norm(newpos, axis=1) > L
    
    for i in range(n):
        if outside[i]:
            #normal vector at wall collision point
            normal_vector = heads[i]/np.linalg.norm(heads[i])
            #reflect the velocity
            v[i] = v[i] - (2 * np.dot(v[i],normal_vector) * normal_vector)
            #updating the position, putting the particle right on the wall
            heads[i] = normal_vector * (L-dL)
        else:
            heads[i] = newpos[i]

        df.loc[len(df)] = ["Ar",heads[i][0],heads[i][1],heads[i][2]]
    snap_add(outfile,df)

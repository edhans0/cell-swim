#!/usr/bin/env python
#Cell Swim Simulation Code
#Version 0: Point Particles with Brownian Motion Only

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

#Force due to LJ potential
def lj(r):
    R = np.linalg.norm(r)
    six = (sigma/R)**6
    twelve = six**2
    force = 48 * eps * 1/r * (twelve - six/2) 
    return(force)

#Run Parameters
n = 10 #number of cells
maxtp = 100
dt = 0.01 #size of time step in s
#total run time = dt * maxtp
mass = 1.
aa = 2.0
b = 1.1 #head-tail distance
#Spherical Simulation Space
L = 50.0 #radius
dL = 0.01*L #thickness (distance from edge at which particles bounce off)
#awall = 1/((L+dL)**4 - L**4) #wall bouncing force

eta = 1.0 #viscosity
pi = np.pi #pi
gamma = 6.0*pi*eta*aa
beta = 1.0 #temperature contral 1/kBT
#constant to multiply random number for stochastic force
co = (2.0/beta/gamma*dt)**0.5

cutoff =10 #determine if potential is turned on or off

#Lennard-Jones Parameters
eps = 0.1
rstar = 3 #LJ cutoff used in Qi2013
sigma = rstar/1.12245295

#Clumping Force Parameters
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

ruv = np.zeros((n,3))
#Initial positions and velocities of each cell
#assigns random positions to each cell within the sphere
#random angles
theta = 2 * np.pi * np.random.rand(n) #azimuthal angle (rotation z-axis)
phi = np.arccos(2*np.random.rand(n) - 1)#polar angle: inclination from positive z-axis
#random distance up to 25 from centre:
R = np.random.rand(n) * L * 0.5
#converting spherical coordinates
heads = np.zeros((n,3))
for i in range(n):
    x = R[i] * np.sin(phi[i]) * np.cos(theta[i])
    y = R[i] * np.sin(phi[i]) * np.sin(theta[i])
    z = R[i] * np.cos(phi[i])
    heads[i] = heads[i] + np.array([x,y,z])
#Initializing velocities to 0
v = np.zeros((n,3))

with open(outfile, 'a') as f:
    f.write(str(n))
    f.write("time, 0")
#write_n_time(outfile, n, 0)

#initial dataframe
cols = ['atomtype','x','y','z','Fsto','F_LJ','Fclump','accel','speed','dr','r']
df = pd.DataFrame(columns=cols)
for i in heads:
    df.loc[len(df)]=['Ar',i[0],i[1],i[2],0,0,0,0,0,0,np.linalg.norm(i)]

snap_add(outfile,df)

for step in range(maxtp):
    t = (step+1)*dt 
    #writing XYZ file headers
    write_n_time(outfile,n,t)
    df = pd.DataFrame(columns=cols)
    for i in range(n):
        Fnet = [np.zeros(3)] #individual contributions (vectors)
        #Stochastic Force
        Fsto = co*0.9*(2*np.random.normal(size=3)-1)
        Fnet.append(Fsto)
        #Lennard Jones
        for j in range(n):
            if i != j:
                r = heads[i]-heads[j]
                F_LJ = lj(r)
                Fnet.append(F_LJ)
        #Clumping Force
        #Vector Sum
        Fnet = sum(Fnet)
        a = Fnet/mass #update acceleration
        dv = a*dt #acceleration * time
        #Update the Velocity
        v[i] = (v[i] + dv)
        #Displacement vector
        dr = v[i] * dt
        #Position Update
        #checks if particle will remain inside sphere. If not, reflect
        newpos = heads[i]+dr
        if np.linalg.norm(newpos) > L:
            #normal vector at wall collision point
            normal_vector = heads[i]/np.linalg.norm(heads[i])
            #reflect the velocity
            v[i] = v[i] - (2 * np.dot(v[i],normal_vector) * normal_vector)
            #updating the position, putting the particle right on the wall
            heads[i] = normal_vector * (L-dL)

        else:
            heads[i] = newpos
        #for checking only
        if np.linalg.norm(heads[i]) > L:
            print(i, np.linalg.norm(heads[i]), "Out of bounds!")
        #output
        #print(i, "Lennard-Jones ", np.linalg.norm(F_LJ))
        #print(i, "Clumping ", np.linalg.norm(Fclump))
        Fsto = np.linalg.norm(Fsto)
        #print("    accel=",np.linalg.norm(a))
        F_LJ = np.linalg.norm(F_LJ)
        Fclump = 0
        df.loc[len(df)] = ["Ar",heads[i][0],heads[i][1],heads[i][2],Fsto,F_LJ,Fclump,np.linalg.norm(a),np.linalg.norm(v[i]),np.linalg.norm(dr),np.linalg.norm(heads[i])]
    snap_add(outfile,df)

#!/usr/bin/env python
#Cell Swim Simulation Code
#Version 1: Point Particles with Active Motion

#imports
import numpy as np
import sys
#Run Parameters
n = int(sys.argv[1]) #number of cells
maxtp = 10000
dt = 0.1 #size of time step in s
#total run time = dt * maxtp
mass = 1.
aa = 2.0
b = 1.1 #head-tail distance
#Spherical Simulation Space
L = 50.0 #radius
dL = 0.1*L #thickness (distance from edge at which particles bounce off)
#awall = 1/((L+dL)**4 - L**4) #wall bouncing force

eta = 1.0 #viscosity
pi = np.pi #pi
gamma = 6.0*pi*eta*aa
beta = 1.0 #temperature contral 1/kBT
#constant to multiply random number for stochastic force
co = (2.0/beta/gamma*dt)**0.5

cutoff =10 #determine if potential is turned on or off

#Lennard-Jones Parameters
eps = 1.0
sigma = 2.0
rstar = sigma*1.12245295 #LJ cutoff

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
#random distance from centre:
R = np.random.rand(n) * (L - dL)
#converting spherical coordinates
heads = np.zeros((n,3))
for i in range(n):
    x = R[i] * np.sin(phi[i]) * np.cos(theta[i])
    y = R[i] * np.sin(phi[i]) * np.sin(theta[i])
    z = R[i] * np.cos(phi[i])
    heads[i] = heads[i] + np.array([x,y,z])
#
print(n, "\n")
for i in heads:
	print("Ar",i[0],i[1],i[2])

for step in range(maxtp):
    #time = dt*step
    v = np.zeros((n,3))
    print(n, "\n")
    for i in range(n):
        Fnet = [np.zeros(3)] #individual contributions (vectors)
        #Potentials
        for j in range(n):
            if i != j:
                #distance
                R = heads[i]-heads[j]
                r = np.linalg.norm(R)
                #don't compute if too far
                #Lennard-Jones
                if r <= rstar:
                    six = (sigma/r)**6
                    twelve = six**2
                    V_LJ = 4*eps*(twelve-six) #potential energy (scalar)
                    #for a potential V(x,y,z), Fx = V(x)/(x/sqrt(x2+y2+z2)
                    F_LJ = [48*(V_LJ*r/R)] #force (vector)
                    Fnet.append(F_LJ)
                #Clumping Force
                if r0 < r < r1:
                    rclump = R - rm
                    Fclump = 4*c2*rclump**3 - 2*c1*rclump**2
                    Fnet.append(Fclump)
        #Stochastic Force
        Fsto = co*0.9*(2*np.random.normal(size=3)-1)
        Fnet.append(Fsto)


        #apply repulsion if outside or close to outside the container

        #Active Motor (random switching on/off/direction)

        #check if the motor has been active or inactive for t_on/t_off seconds
        #if yes then reroll to determine switch
        #random direction
        if on_state[i] == 0:
            if counter[i] == t_off/dt:
                on_state[i] =  1 #switch
                ruv[i] = (2*np.random.rand(3)-1)/np.linalg.norm(np.random.rand(3)-1)
                n_hat = on_state[i] * ruv[i] #direction unit vector
                counter[i] = 0
            else:
                counter[i] = counter[i] + 1 #increase the counter
                n_hat = 0
        else:
            if counter[i] == t_on/dt:
                on_state[i] = np.random.randint(0,2) #switch
                randomvector = 2*np.random.rand(3)-1
                ruv[i] = (2*np.random.rand(3)-1)/np.linalg.norm(2*np.random.rand(3)-1)
                n_hat = on_state[i] * ruv[i]
                counter[i] = 0 #reset the counter

            #increase the counter
            else:
                n_hat = on_state[i] * ruv[i]
                counter[i] = counter[i] + 1
        ##If particle is currently in a cluster, turn off the motor
        #if r0<r<r1:
            #n_hat = 0
        #Vector Sum
        Fnet = sum(Fnet)
        a = Fnet/mass #update acceleration
        dv = a*dt #acceleration * time
        #Update the Velocity
        v[i] = (v[i] + dv + (n_hat*vm))
        #Displacement vector
        dr = v[i] * dt
        #Position Update
        #checks if particle will remain inside sphere. If not, reflect
        newpos = heads[i]+dr
        if np.linalg.norm(newpos) >= L-dL:
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
        print("Ar",heads[i][0],heads[i][1],heads[i][2])
        #print(i, "n_hat=",n_hat, "velocity=",v[i])
        #print(bool(on_state[i]), counter[i], np.linalg.norm(v[i]))

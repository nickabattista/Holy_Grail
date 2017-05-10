 2D LATTICE BOLTZMANN (LBM) SIMULATION 
Author: Nicholas A. Battista
Created: 11/4/2014  (MATLAB)
Modified: 12/2/2014 (MATLAB)
Created: 05/05/2017 (Python3)

_____________________________________________________________________________

D2Q9 Model:

c6  c2   c5
  \  |  /  
c3- c9 - c1
  /  |  \  
c7  c4   c8

f_i: the probability for site vec(x) to have a particle heading in
direction i, at time t. These f_i's are called discretized probability 
distribution functions

_____________________________________________________________________________

LBM Idea: 
1. At each timestep the particle densities propogate in each direction (1-8).
2. An equivalent "equilibrium" density is found
3. Densities relax towards that state, in proportion governed by tau
   (parameter related to viscosity)

_____________________________________________________________________________

CHOICE OF SIMULATION:
-The code is setup to run a few different geometries:
	a. Flow in a channel
	b. Flow around a cylinder
	c. Flow around a few cylinders
	d. Flow through one porous layer
	e. Flow through multiple porous layers

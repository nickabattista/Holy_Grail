'''
 2D LATTICE BOLTZMANN (LBM) SIMULATION 
 Author: Nicholas A. Battista
 Created: 11/4/2014  (MATLAB)
 Modified: 12/2/2014 (MATLAB)
 Created: 5/5/2017   (Python3)

  D2Q9 Model:
  c6  c2   c5  
    \  |  /    
  c3- c9 - c1  
    /  |  \   
  c7  c4   c8     

f_i: the probability for site vec(x) to have a particle heading in
     direction i, at time t. These called discretized probability
     distribution functions and represent the central link to LBMs.

LBM Idea: 
         1. At each timestep the particle densities propogate in each direction (1-8). 
         2. An equivalent "equilibrium' density is found
         3. Densities relax towards that state, in proportion governed by
            tau (parameter related to viscosity).
'''

import numpy as np
import numpy.matlib as matlib
from scipy import misc, fftpack
import os
from print_vtk_files import *

###########################################################################
#
# Function to print Lattice Boltzmann key ideas to screen.
#
###########################################################################

def print_LBM_Info():

    print('\n\n 2D LATTICE BOLTZMANN (LBM) SIMULATION \n')
    print('Author: Nicholas A. Battista\n')
    print('Created: 11/4/2014   (MATLAB)\n')
    print('Modified: 12/2/2014  (MATLAB)\n')
    print('Created: 5/5/2017    (Python3)\n\n')
    print('_____________________________________________________________________________\n\n')
    print('D2Q9 Model:\n\n')
    print('c6  c2   c5\n')
    print('  \\  |  /  \n')
    print('c3- c9 - c1\n')
    print('  /  |  \\  \n')
    print('c7  c4   c8\n\n')

    print('f_i: the probability for site vec(x) to have a particle heading in\n')
    print('direction i, at time t. These f_i''s are called discretized probability \n')
    print('distribution functions\n\n')

    print('LBM Idea: \n')
    print('1. At each timestep the particle densities propogate in each direction (1-8).\n')
    print('2. An equivalent "equilibrium" density is found\n')
    print('3. Densities relax towards that state, in proportion governed by tau\n')
    print('#s (parameter related to viscosity)\n\n','  ')
    print('_____________________________________________________________________________\n\n')


###########################################################################
#
# Function to print specific simulation info to screen
#
###########################################################################

def print_simulation_info(choice):

    if (choice=='channel'):

        print('You are simulating CHANNEL FLOW\n')
        print('Flow proceeds left to right through the channel\n')
        print('You should see a parabolic flow profile develop\n\n\n')

    elif (choice=='cylinder1'):

        print('You are simulating flow around a cylinder\n')
        print('Flow proceeds left to right through the channel containing a 2D cylinder\n')
        print('You should see flow wrapping around the cylinder\n')
        print('Try changing the tau (viscosity) to observe differing dynamics\n\n\n')

    elif (choice=='cylinder2'):

        print('You are simulating flow around a field of cylinders\n')
        print('Flow proceeds left to right through the channel containing a 2D cylinder\n')
        print('You should see flow wrapping around the cylinders\n')
        print('Try changing the tau (viscosity) to observe differing dynamics\n')
        print('Also try adding cylinders or changing their place in the "give_Me_Problem_Geometry" function\n\n\n')

    elif (choice=='porous1'):

        print('You are simulating porous media flow\n')
        print('Flow proceeds left to right through the channel containing obstacles\n')
        print('Try changing the porosity (percentPorosity) to observe differing dynamics\n')
        print('NOTE: each simulation creates a RANDOM porous geometry\n\n\n')

    elif (choice=='porous2'):

        print('You are simulating flow through various porous layers\n')
        print('Flow proceeds left to right through the channel containing obstacles\n')
        print('Try changing the porosity (percentPorosity) to observe differing dynamics\n')
        print('NOTE: each simulation creates a RANDOM porous geometry\n\n\n')


##########################################################################
#
# Function to choose what geometry to consider for the simulation
# Returns: Geometry / Increase to inlet velocity for each time-step / endTime
#
##########################################################################

def give_Me_Problem_Geometry(choice,nx,ny,percentPorosity):


    if (choice=='channel'):

        #CHANNEL GEOMETRY
        BOUNDs=np.zeros((nx,ny))
        BOUNDs[0:,[0,ny-1]]=1.0            # PUTS "1's" on LEFT/RIGHT Boundaries
        deltaU = 0.01                      # Incremental increase to inlet velocity
        endTime = 2500

    '''
    elif (choice=='cylinder1'):

        #CHANNEL FLOW W/ CYLINDER
        a=repmat(-(nx-1)/2:(nx-1)/2,[ny,1]) 
        r = floor(nx/5)
        aR = ceil(nx/5)
        BOUND=( a.^2+(a+aR)'.^2)<r         #PUTS "1's" within region of Cylinder
        BOUND(1:nx,[1 ny])=1               #Puts "1's" on Left/Right Boundaries
        deltaU = 0.01                      #Incremental increase to inlet velocity
        endTime = 2500

    elif (choice=='cylinder2'):

        #CHANNEL FLOW W/ CYLINDER
        a=repmat(-(nx-1)/2:(nx-1)/2,[ny,1])
        r = floor(nx/2.5)
        aL = floor(nx/5)
        aR = ceil(nx/2)
        aM = 0.3*aR
        B1=  ( ( (a).^2+(a+3.75*aR/5)'.^2)<r )           #PUTS "1's" within region of Cylinder1
        B2=  ( ( (a+aL/1.75).^2+(a+4*aM/5)'.^2)<r )        #PUTS "1's" within region of Cylinder2
        B3=  ( ( (a-aL/1.75).^2+(a+4*aM/5)'.^2)<r )        #PUTS "1's" within region of Cylinder1
        BOUND= double(B1)+double(B2)+double(B3) #PUTS together all cylinder geometry
        BOUND(1:nx,[1 ny])=1                    #Puts "1's" on Left/Right Boundaries
        deltaU = 0.01                           #Incremental increase to inlet velocity
        endTime = 5000


    elif (choice=='porous1'):

        #POROUS RANDOM DOMAIN
        BOUND=rand(nx,ny)<percentPorosity   #PUTS "1's" inside domain randomly if RAND value above percent  
        aS = ceil(nx/5)
        aE = ceil(4*5/nx)
        BOUND(1:aS,:) = 0 
        BOUND(aE:end,:)=0
        BOUND(1:nx,[1 ny])=1                #PUTS "1's" on LEFT/RIGHT Boundaries
        deltaU = 1e-7                       #Incremental increase to inlet velocity
        endTime = 5000

    elif (choice=='porous2'):

        #POROUS RANDOM DOMAIN
        BOUND=rand(nx,ny)<percentPorosity  #PUTS "1's" inside domain randomly if RAND value above percent              
        BOUND(1:floor(9*nx/31),:) = 0                   #PUTS "0's" to make open channels through porous structure
        BOUND(floor(7*nx/31):floor(9*nx/31),:) = 0                   #PUTS "0's" to make open channels through porous structure
        BOUND(floor(13*nx/31):floor(15*nx/31),:) = 0                 #PUTS "0's" to make open channels through porous structure
        BOUND(floor(19/31*nx):floor(21/31*nx),:) = 0                 #PUTS "0's" to make open channels through porous structure
        BOUND(floor(25/31*nx):floor(27/31*nx),:)=0                   #PUTS "0's" to make open channels through porous structure
        BOUND(floor(30/31*nx):end,:) = 0                #PUTS "0's" to make open channels through porous structure
        BOUND(1:nx,[1 ny])=1               #PUTS "1's" on LEFT/RIGHT Boundaries
        deltaU = 1e-7                      #Incremental increase to inlet velocity
        endTime = 5000
    '''

    return BOUNDs, deltaU, endTime


##########################################################################
#
# Function to stream the distribution function, f.
#
##########################################################################

def please_Stream_Distribution(f,nx,ny):
    
    a1=np.array([nx-1])
    a2=np.arange(0,nx-1,1)
    inds = np.concatenate((a1, a2), axis=0)
    f[0,:,:] =f[0,inds,:]                 #Stream Right

    a1=np.array([ny-1])
    a2=np.arange(0,ny-1,1)
    inds = np.concatenate((a1, a2), axis=0)
    f[1,:,:] =f[1,:,inds]                 #Stream Up

    a1=np.array([0])
    a2=np.arange(1,nx,1)
    inds = np.concatenate((a2, a1), axis=0)
    f[2,:,:] =f[2,inds,:]                #Stream Left

    a1=np.array([0])
    a2=np.arange(1,ny,1)
    inds = np.concatenate((a2, a1), axis=0)
    f[3,:,:] =f[3,:,inds]                #Stream Down

    a1=np.array([nx-1]); a2=np.arange(0,nx-1,1)
    inds = np.concatenate((a1, a2), axis=0)
    b1=np.array([ny-1]); b2=np.arange(0,ny-1,1)
    indsY = np.concatenate((b1, b2), axis=0)
    f[4,:,:] =f[4,inds,indsY]           #Stream Right-Up

    a1=np.array([0]); a2=np.arange(1,nx,1)
    inds = np.concatenate((a2, a1), axis=0)
    b1=np.array([ny-1]); b2=np.arange(0,ny-1,1)
    indsY = np.concatenate((b1, b2), axis=0)
    f[5,:,:] =f[5,inds,indsY]           #Stream Left-Up

    a1=np.array([0]); a2=np.arange(1,nx,1)
    inds = np.concatenate((a2, a1), axis=0)
    b1=np.array([0]); b2=np.arange(1,ny,1)
    indsY = np.concatenate((b2, b1), axis=0)
    f[6,:,:] =f[6,inds,indsY]           #Stream Left-Down    

    a1=np.array([nx-1]); a2=np.arange(0,nx-1,1)
    inds = np.concatenate((a1, a2), axis=0)
    b1=np.array([0]); b2=np.arange(1,ny,1)
    indsY = np.concatenate((b2, b1), axis=0)
    f[7,:,:] =f[7,inds,indsY]           #Stream Right-Down

    return f


##########################################################################
#
# Function to give the equilibrium distribution, f_EQ.
#
##########################################################################

def please_Give_Equilibrium_Distribution(w1,w2,w3,DENSITY,UX,UY,U_SQU,U_5,U_6,U_7,U_8,f_EQ):
     
    # Calculate equilibrium distribution: stationary pt in middle.
    f_EQ[8,:,:] = w1*DENSITY * (1 - (3/2)*U_SQU )
    
   
    # NEAREST-neighbours (i.e., stencil pts directly right,left,top,bottom)
    # Equilibrium DFs can be obtained from the local Maxwell-Boltzmann SPDF 
    f_EQ[0,:,:] = w2*DENSITY * (1 + 3*UX + (9/2)*(UX*UX) - (3/2)*U_SQU )
    f_EQ[1,:,:] = w2*DENSITY * (1 + 3*UY + (9/2)*(UY*UY) - (3/2)*U_SQU )
    f_EQ[2,:,:] = w2*DENSITY * (1 - 3*UX + (9/2)*(UX*UX) - (3/2)*U_SQU )
    f_EQ[3,:,:] = w2*DENSITY * (1 - 3*UY + (9/2)*(UY*UY) - (3/2)*U_SQU )
    
    # NEXT-NEAREST neighbours (i.e., diagonal elements for stencil pts)
    # Equilibrium DFs can be obtained from the local Maxwell-Boltzmann SPDF 
    f_EQ[4,:,:] = w3*DENSITY * (1 + 3*U_5 + (9/2)*(U_5*U_5) - (3/2)*U_SQU )
    f_EQ[5,:,:] = w3*DENSITY * (1 + 3*U_6 + (9/2)*(U_6*U_6) - (3/2)*U_SQU )
    f_EQ[6,:,:] = w3*DENSITY * (1 + 3*U_7 + (9/2)*(U_7*U_7) - (3/2)*U_SQU )
    f_EQ[7,:,:] = w3*DENSITY * (1 + 3*U_8 + (9/2)*(U_8*U_8) - (3/2)*U_SQU )
    
    return f_EQ





###########################################################################
#
# Function that performs the SIMULATION!
#
###########################################################################

def lets_do_LBM():

    # 2D LATTICE BOLTZMANN (LBM) SIMULATION 
    # Author: Nicholas A. Battista
    # Created: 11/4/2014
    # Modified: 12/2/2014

    #  D2Q9 Model:
    #  c6  c2   c5  
    #    \  |  /    
    #  c3- c9 - c1  
    #    /  |  \   
    #  c7  c4   c8     

    #f_i: the probability for site vec(x) to have a particle heading in
    #     direction i, at time t. These called discretized probability
    #     distribution functions and represent the central link to LBMs.

    #LBM Idea: 
    #         1. At each timestep the particle densities propogate in each direction (1-8). 
    #         2. An equivalent "equilibrium' density is found
    #         3. Densities relax towards that state, in proportion governed by
    #            tau (parameter related to viscosity).


    #Prints key ideas to screen
    print_LBM_Info() 

    #
    # Simulation Parameters
    #
    tau=0.53                               # Tau: relaxation parameter related to viscosity
    density=0.01                           # Density to be used for initializing whole grid to value 1.0
    w1=4/9; w2=1/9; w3=1/36                # Weights for finding equilibrium distribution
    nx=320; ny=320                         # Number of grid cells
    nx=10; ny=10                           # Number of grid cells
    Lx = 1; Ly = 1                         # Size of computational domain
    dx = Lx/nx; dy = Ly/ny                 # Grid Resolutions in x and y directions, respectively
    f = density/9.0*np.ones((9,nx,ny))     # Copies density/9 into 9-matrices of size [nx,ny] -> ALLOCATION for all "DIRECTIONS"
    f_EQ = f                               # Initializes F-equilibrium Storage space
    grid_size= nx*ny                       # Total number of grid cells
    CI= np.arange(0,8*grid_size,grid_size) # Indices to point to FIRST entry of the desired "z-stack" distribution grid      


    #
    # Chooses which problem to simulate
    #
    # Possible Choices: 'cylinder1', 'cylinder2', 'channel', 'porous1', 'porous2'
    #
    choice = 'channel'
    percentPorosity = 0.25  # Percent of Domain that's Porous (does not matter if not studying porous problem)
    BOUND, deltaU, endTime = give_Me_Problem_Geometry(choice,nx,ny,percentPorosity) #BOUND: gives geometry, deltaU: gives incremental increase to inlet velocity
    print_simulation_info(choice)


    #Find Indices of NONZERO Elements, i.e., where "boundary points" are
    ON_i,ON_j = np.nonzero(BOUND) # matrix offset of each Occupied Node e.g., A(ON_i,ON_j) ~= 0
    print(BOUND)
    print(BOUND[ON_i,ON_j])

    A=zeros(1,2)

    #Offsets Indices for the Different Directions [i.e., levels of F_i=F(:,:,i) ] for known BOUNDARY pts.
    #TO_REFLECT=[ON+CI(1) ON+CI(2) ON+CI(3) ON+CI(4) ON+CI(5) ON+CI(6) ON+CI(7) ON+CI(8)]
    #REFLECTED= [ON+CI(3) ON+CI(4) ON+CI(1) ON+CI(2) ON+CI(7) ON+CI(8) ON+CI(5) ON+CI(6)]


    #Initialization Parameters
    #avgU=1                           #initialize avg. velocity to 1.0
    #prevAvgU=1                       #initialize previous-avg. velocity to 1.0
    ts=0                             #initialize starting time to 0
    #numactivenodes=sum(sum(1-BOUND)) #Finds number of nodes that ARE NOT boundary pts.


    # SAVING DATA TO VTK #
    print_dump = int(np.floor(endTime/50))
    ctsave = 0
    # CREATE VTK DATA FOLDER
    try:
        os.mkdir('vtk_data_test')
    except FileExistsError:
        #File already exists
        pass

    # INITIALIZE DATA STORAGE #    
    UX = np.zeros((nx,ny))                                # Initialize x-Component of Velocity
    UY = np.zeros((nx,ny))                                # Initialize y-Component of Velocity
    vorticity = np.zeros((nx-1,ny-1))                     # Initialize Vorticity   
    print_vtk_files(ctsave,UX,UY,vorticity,Lx,Ly,nx,ny)   # Print .vtk files for initial configuration


    #Begin time-stepping!
    while ( ts < endTime ):

        # STREAMING STEP (progate in respective directions)
        f = please_Stream_Distribution(f,nx,ny)

       
        #Densities bouncing back at next timestep
        #BOUNCEDBACK=f(TO_REFLECT) 

        #vec(rho) = SUM_i f_i -> SUMS EACH DISTRIBUTION MATRIX TOGETHER
        DENSITY=sum(f)  # Note: denotes sum over third dimension

        #vec(u) = 1/vec(rho) SUM_i (f_i)(e_i) -> CREATES VELOCITY MATRICES
        UX = ( ( f[0,:,:] + f[4,:,:] + f[7,:,:] ) - ( f[2,:,:] + f[5,:,:] + f[6,:,:]) ) / DENSITY
        # MATLAB: UX=( sum(f(:,:,[1 5 8]),3)-sum(f(:,:,[3 6 7]),3) ) ./ DENSITY 
        UY = ( ( f[1,:,:] + f[4,:,:] + f[5,:,:] ) - ( f[3,:,:] + f[6,:,:] + f[7,:,:]) ) / DENSITY
        # MATLAB: UY=( sum(f(:,:,[2 5 6]),3)-sum(f(:,:,[4 7 8]),3) ) ./ DENSITY

        #Increase inlet velocity with each time step along left wall
        UX[0,:] = UX[0,:] + deltaU 

        #Enforce BCs to Zero Velocity / Zero Density
        UX[ON_i,ON_j] = 0      # Makes all Boundary Regions have zero x-velocity -> MATLAB: UX(ON)=0      
        UY[ON_i,ON_j] = 0      # Makes all Boundary Regions have zero y-velocity -> MATLAB: UY(ON)=0
        DENSITY[ON_i,ON_j] = 0 # Makes DENSITY of Boundary Regions have zero value -> MATLAB: DENSITY(ON)=0

        #Square of Magnitude of Velocity Overall
        U_SQU = UX*UX + UY*UY 

        #Create "Diagonal" Velocity Quantities
        U_5 =  UX+UY #Create velocity direction to Point 5
        U_6 = -UX+UY #Create velocity direction to Point 6
        U_7 = -U_5   #Create velocity direction to Point 7
        U_8 = -U_6   #Create velocity direction to Point 8

        #Calculate the equilibrium distribution
        f_EQ = please_Give_Equilibrium_Distribution(w1,w2,w3,DENSITY,UX,UY,U_SQU,U_5,U_6,U_7,U_8,f_EQ)

        #Update the PDFs
        f = f - (1/tau)*(f-f_EQ)

        #BOUNCE BACK DENSITIES for next time-step
        #f(REFLECTED)= BOUNCEDBACK

        #Updates simulation parameters
        ts=ts+1   # update time step
        
        # Save files info!
        ctsave = ctsave + 1
        if (ctsave % print_dump) == 0:

            # compute vorticity
            dUx_y = ( UX[0:nx-2,1:ny-1] - UX[0:nx-2,0:ny-2] ) / dy
            dUy_x = ( UY[1:nx-1,0:ny-2] - UY[0:nx-2,0:ny-2] ) / dx
            vorticity[0:nx-2,0:ny-2] = dUy_x - dUx_y 

            # print to vtk
            print_vtk_files(ctsave,UX,UY,vorticity,Lx,Ly,nx,ny)
            fprintf('Simulation Time: #d\n',ts)
        


if __name__ == "__main__":
    lets_do_LBM()    
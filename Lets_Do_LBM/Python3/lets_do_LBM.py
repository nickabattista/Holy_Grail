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
import time

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
    print('_____________________________________________________________________________\n\n')
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
        print('Flow proceeds left to right through the channel containing 2D cylinders\n')
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


###########################################################################
#
# Function to give boundary points for visualization in vtk printing.
#
###########################################################################
    
def give_Me_Boundary_Pts_For_Visualization(dx,dy,Nx,Ny,Lx,Ly,ON_i,ON_j):

    aX = np.arange(Lx-dx,-dx,-dx)
    xMat = np.tile(aX,[Ny,1])           # MATLAB: xMat = repmat(Lx-dx:-dx:0,Nx).T

    aY = np.arange(0,Ly,dy)
    yMat = np.tile(aY,[Nx,1]).T         # MATLAB: yMat = repmat(0:dy:Ly,Ny)
    
    xBounds = xMat[ON_i,ON_j]
    xBounds = -xBounds + Lx
    yBounds = yMat[ON_i,ON_j]



    lenX = len(xBounds)
    Bound_Pts = np.zeros((lenX,2))
    
    for i in range(0,lenX):
        Bound_Pts[i,0] = xBounds[i]
        Bound_Pts[i,1] = yBounds[i]
    
    # TEST PLOTTING IN MATLAB -> CONVERT TO PYTHON ONE DAY
    #subplot(1,2,1)
    #image(2-BOUND') hold on
    #subplot(1,2,2)
    #plot(Bound_Pts(:,1),Bound_Pts(:,2),'*')
    #pause() 
    
    return Bound_Pts



##########################################################################
#
# Function to choose what geometry to consider for the simulation
# Returns: Geometry / Increase to inlet velocity for each time-step / endTime
#
##########################################################################

def give_Me_Problem_Geometry(choice,Nx,Ny,percentPorosity):


    if (choice=='channel'):

        #CHANNEL GEOMETRY
        BOUNDs=np.zeros((Nx,Ny))
        BOUNDs[0:,[0,Ny-1]]=1.0            # PUTS "1's" on LEFT/RIGHT Boundaries
        deltaU = 0.01                      # Incremental increase to inlet velocity
        endTime = 2500

    elif (choice=='porous1'):

        #POROUS RANDOM DOMAIN
        BOUNDs=np.random.rand(Nx,Ny)<(1-percentPorosity)   # Puts TRUE/FALSE's inside domain randomly if RAND value above percent
        BOUNDs=BOUNDs.astype(int)                          # Converts Boolean matrix to matrix of 0,1's  
        aS = int(np.ceil(Nx/5))                            # Puts openings near ends of porous channel
        aE = int(np.ceil(4*Nx/5))                          # Puts openings near ends of porous channel
        BOUNDs[0:aS,:] = 0                                 # Puts openings near ends of porous channel
        BOUNDs[aE:,:]=0                                    # Puts openings near ends of porous channel
        BOUNDs[0:,[0,Ny-1]]=1                              # Puts "1's" on LEFT/RIGHT Boundaries
        deltaU = 1e-7                                      # Incremental increase to inlet velocity
        endTime = 5000

    elif (choice=='porous2'):

        #POROUS RANDOM DOMAIN
        BOUNDs=np.random.rand(Nx,Ny)<(1-percentPorosity)                 # PUTS "1's" inside domain randomly if RAND value above percent              
        BOUNDs[np.arange(0,int(np.floor(9*Nx/31)),1),:] = 0                        # PUTS "0's" to make open channels through porous structure
        BOUNDs[np.arange(int(np.floor(7*Nx/31)),int(np.ceil(9*Nx/31)),1),:] = 0    # PUTS "0's" to make open channels through porous structure
        BOUNDs[np.arange(int(np.floor(13*Nx/31)),int(np.ceil(15*Nx/31)),1),:] = 0  # PUTS "0's" to make open channels through porous structure
        BOUNDs[np.arange(int(np.floor(19/31*Nx)),int(np.ceil(21/31*Nx)),1),:] = 0  # PUTS "0's" to make open channels through porous structure
        BOUNDs[np.arange(int(np.floor(25/31*Nx)),int(np.ceil(27/31*Nx)),1),:]=0    # PUTS "0's" to make open channels through porous structure
        BOUNDs[int(np.floor(30/31*Nx)):,:] = 0                                     # PUTS "0's" to make open channels through porous structure
        BOUNDs[0:,[0,Ny-1]]= 1                                           # PUTS "1's" on LEFT/RIGHT Boundaries
        deltaU = 1e-7                                                    # Incremental increase to inlet velocity
        endTime = 5000
    

    elif (choice=='cylinder1'):

        #CHANNEL FLOW W/ CYLINDER

        # radii of cylinder (given in terms of mesh widths)
        radius1 = 0.075*Ny

        # auxillary vectors 
        xAux = np.arange(-(Nx-1)/2, (Nx-1)/2+1, 1)
        yAux = np.arange(-(Ny-1)/2, (Ny-1)/2+1, 1)

        # stack vectors to create grids of indices
        a1=np.tile(xAux,(Ny,1))
        a2=np.tile(yAux,(Nx,1))          

        # Amount to translate cylinder from middle of domain
        aR = np.floor(0.375*Nx)    

        # Create Cylinder Geometry (Note: "+" shifts left bc of defining circles)
        BOUNDs= ( (a1+aR)**2+((a2).T)**2) < radius1**2     # Puts "1's" within Cylinder Region

        # Convert to 0,1 boolean matrix from False,True matrix
        BOUNDs = BOUNDs.astype(int)

        # Create Top/Bottom Boundaries
        BOUNDs[[0,Ny-1],0:] = 1              # Puts "1's" on Top/Bottom Boundaries

        deltaU = 0.00125                     # Incremental increase to inlet velocity
        endTime = 56000                      # Total Number of Time-Steps

    
    elif (choice=='cylinder2'):

        #CHANNEL FLOW W/ MULTIPLE CYLINDERS
        
        # radii of cylinder (given in terms of mesh widths)
        radius1 = 0.125*Ny

        # auxillary vectors 
        xAux = np.arange(-(Nx-1)/2, (Nx-1)/2+1, 1)
        yAux = np.arange(-(Ny-1)/2, (Ny-1)/2+1, 1)

        # stack vectors to create grids of indices
        a1=np.tile(xAux,(Ny,1))
        a2=np.tile(yAux,(Nx,1))          

        # Amount to translate cylinder from middle of domain
        aR = np.floor(0.375*Nx)    
        aSx = np.floor(0.1*Nx)
        aY = np.floor(0.2*Ny)
    
        # Create Cylinder Geometry (Note: "+" shifts left bc of defining circles)
        B1 = ( (a1+aR)**2+((a2).T)**2) < radius1**2            # Puts "1's" within Cylinder Region
        B2 = ( (a1+aR-aSx)**2+((a2-aY).T)**2) < radius1**2     # Puts "1's" within Cylinder Region
        B3 = ( (a1+aR-aSx)**2+((a2+aY).T)**2) < radius1**2     # Puts "1's" within Cylinder Region

        # Convert to 0,1 boolean matrix from False,True matrix
        B1 = B1.astype(int)
        B2 = B2.astype(int)
        B3 = B3.astype(int)

        # Combine boolean matrices
        # Note: here assuming no overlapping vorticity regions
        BOUNDs = B1 + B2 + B3

        # Create Top/Bottom Boundaries
        BOUNDs[[0,Ny-1],0:] = 1              # Puts "1's" on Top/Bottom Boundaries

        deltaU = 0.01                     # Incremental increase to inlet velocity
        endTime = 5500                    # Total Number of Time-Steps


    #
    # Reverse Order of BOUND for GEOMETRY (top is initialized as bottom and vice versa)
    #
    Bound2 = np.zeros( BOUNDs.shape )
    for i in range(0,Ny):
        Bound2[i,0:] = BOUNDs[Ny-1-i,0:]

    return BOUNDs, Bound2, deltaU, endTime


##########################################################################
#
# Function to stream the distribution function, f.
#
##########################################################################

def please_Stream_Distribution(f,Nx,Ny):
    
    xInds = np.arange(0,Nx,1)                # Indices Vector 0,1,...,Ny-1
    yInds = np.arange(0,Ny,1)                # Indices Vector 0,1,...,Nx-1


    # STREAM RIGHT
    a1=np.array([Nx-1])
    a2=np.arange(0,Nx-1,1)
    inds = np.concatenate((a1, a2), axis=0)
    f[0,xInds,:] =f[0,inds,:]                 #Stream Right


    # STREAM UP
    a1=np.array([Ny-1])
    a2=np.arange(0,Ny-1,1)
    inds = np.concatenate((a1, a2), axis=0)
    f[1,:,yInds] =f[1,:,inds]                 #Stream Up


    # STREAM LEFT
    a1=np.array([0])
    a2=np.arange(1,Nx,1)
    inds = np.concatenate((a2, a1), axis=0)
    f[2,xInds,:] =f[2,inds,:]                #Stream Left


    # STREAM DOWN
    a1=np.array([0])
    a2=np.arange(1,Ny,1)
    inds = np.concatenate((a2, a1), axis=0)
    f[3,:,yInds] =f[3,:,inds]                #Stream Down


    # STREAM RIGHT-UP!
    a1=np.array([Nx-1]); a2=np.arange(0,Nx-1,1)
    inds = np.concatenate((a1, a2), axis=0)
    b1=np.array([Ny-1]); b2=np.arange(0,Ny-1,1)
    indsY = np.concatenate((b1, b2), axis=0)
    f[4,xInds,:] =f[4,inds,:]            #Stream Right
    f[4,:,yInds] =f[4,:,indsY]           #Stream Up


    # STREAM LEFT-UP!
    a1=np.array([0]); a2=np.arange(1,Nx,1)
    inds = np.concatenate((a2, a1), axis=0)
    b1=np.array([Ny-1]); b2=np.arange(0,Ny-1,1)
    indsY = np.concatenate((b1, b2), axis=0)
    f[5,xInds,:] =f[5,inds,:]            #Stream Left
    f[5,:,yInds] =f[5,:,indsY]           #Stream Up


    # STREAM LEFT-DOWN
    a1=np.array([0]); a2=np.arange(1,Nx,1)
    inds = np.concatenate((a2, a1), axis=0)
    b1=np.array([0]); b2=np.arange(1,Ny,1)
    indsY = np.concatenate((b2, b1), axis=0)
    f[6,xInds,:] =f[6,inds,:]            #Stream Left
    f[6,:,yInds] =f[6,:,indsY]           #Stream Down


    # STREAM RIGHT-DOWN
    a1=np.array([Nx-1]); a2=np.arange(0,Nx-1,1)
    inds = np.concatenate((a1, a2), axis=0)
    b1=np.array([0]); b2=np.arange(1,Ny,1)
    indsY = np.concatenate((b2, b1), axis=0)
    f[7,xInds,:] =f[7,inds,:]            #Stream Right
    f[7,:,yInds] =f[7,:,indsY]           #Stream Down


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
    Nx=640; Ny=160                         # Number of grid cells
    Lx = 2; Ly = 0.5                       # Size of computational domain
    dx = Lx/Nx; dy = Ly/Ny                 # Grid Resolutions in x and y directions, respectively
    

    #
    # Chooses which problem to simulate
    #
    # Possible Choices: 'cylinder1', 'cylinder2', 'porousCylinder
    #
    choice = 'cylinder2'
    percentPorosity = 0.625  # Percent of Domain that's Porous (does not matter if not studying porous problem)
    BOUND, Bound2, deltaU, endTime = give_Me_Problem_Geometry(choice,Nx,Ny,percentPorosity) #BOUND: gives geometry, deltaU: gives incremental increase to inlet velocity
    print_simulation_info(choice)


    #
    # Gridding Initialization
    #
    f = density/9.0*np.ones((9,Nx,Ny))     # Copies density/9 into 9-matrices of size [Nx,Ny] -> ALLOCATION for all "DIRECTIONS"
    f_EQ = density/9.0*np.ones((9,Nx,Ny))  # Initializes F-equilibrium Storage space
    grid_size= Nx*Ny                       # Total number of grid cells
    CI= np.arange(0,8*grid_size,grid_size) # Indices to point to FIRST entry of the desired "z-stack" distribution grid      
    


    #Find Indices of NONZERO Elements, i.e., where "boundary points" are
    ON_i,ON_j = np.nonzero(BOUND.T)       # matrix offset of each Occupied Node e.g., A(ON_i,ON_j) ~= 0
    On2_i,On2_j = np.nonzero(Bound2)      # matrix index of each occupied node for visualization data
    BOUNCEDBACK = np.zeros((ON_i.size,8)) 

    # Give Boundary Points For Saving Data
    Bound_Pts = give_Me_Boundary_Pts_For_Visualization(dx,dy,Nx,Ny,Lx,Ly,On2_i,On2_j)

    #Initialization Parameters
    ts=0                              # initialize starting time to 0 (time-step)
    print('Simulation Time: {0:6.6f}\n'.format(ts))


    # SAVING DATA TO VTK #
    print_dump = 100 #int(np.floor(endTime/50))  # number of time-steps between data storage
    ctsave = 0                              # Counts # of total time-steps
    pSave = 0                               # Counts total number of time-steps with data saved
    
    # CREATE VTK DATA FOLDER
    try:
        os.mkdir('vtk_data')
    except FileExistsError:
        #File already exists
        pass

    # INITIALIZE DATA STORAGE #    
    UX = np.zeros((Nx,Ny))                                          # Initialize x-Component of Velocity
    UY = np.zeros((Nx,Ny))                                          # Initialize y-Component of Velocity
    vorticity = np.zeros((Nx-1,Ny-1))                               # Initialize Vorticity   
    print_vtk_files(pSave,UX,UY,vorticity,Lx,Ly,Nx,Ny,Bound_Pts)   # Print .vtk files for initial configuration


    #Begin time-stepping!
    while ( ts < endTime ):


        # STREAMING STEP (progate in respective directions)
        f = please_Stream_Distribution(f,Nx,Ny)


        #Densities bouncing back at next timestep
        #BOUNCEDBACK=f(TO_REFLECT) 
        BOUNCEDBACK[:,0] = f[0,ON_i,ON_j]
        BOUNCEDBACK[:,1] = f[1,ON_i,ON_j]
        BOUNCEDBACK[:,2] = f[2,ON_i,ON_j]
        BOUNCEDBACK[:,3] = f[3,ON_i,ON_j]
        BOUNCEDBACK[:,4] = f[4,ON_i,ON_j]
        BOUNCEDBACK[:,5] = f[5,ON_i,ON_j]
        BOUNCEDBACK[:,6] = f[6,ON_i,ON_j]
        BOUNCEDBACK[:,7] = f[7,ON_i,ON_j]
        
        #time.sleep(2.0) 

        #vec(rho) = SUM_i f_i -> SUMS EACH DISTRIBUTION MATRIX TOGETHER
        DENSITY=sum(f)  # Note: denotes sum over third dimension

        #vec(u) = 1/vec(rho) SUM_i (f_i)(e_i) -> CREATES VELOCITY MATRICES
        UX = ( ( f[0,:,:] + f[4,:,:] + f[7,:,:] ) - ( f[2,:,:] + f[5,:,:] + f[6,:,:]) ) / DENSITY
        # MATLAB: UX=( sum(f(:,:,[1 5 8]),3)-sum(f(:,:,[3 6 7]),3) ) ./ DENSITY 
        UY = ( ( f[1,:,:] + f[4,:,:] + f[5,:,:] ) - ( f[3,:,:] + f[6,:,:] + f[7,:,:]) ) / DENSITY
        # MATLAB: UY=( sum(f(:,:,[2 5 6]),3)-sum(f(:,:,[4 7 8]),3) ) ./ DENSITY

        #Increase inlet velocity with each time step along left wall
        UX[0,1:Ny] = UX[0,1:Ny] + deltaU 


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
        #f(REFLECTED) = BOUNCEDBACK
        #REFLECTED= [ON+CI(3) ON+CI(4) ON+CI(1) ON+CI(2) ON+CI(7) ON+CI(8) ON+CI(5) ON+CI(6)]
        #REFLECTED= [   CI(3)     CI(4)     CI(1)     CI(2)     CI(7)     CI(8)     CI(5)     CI(6)  ]
        #REFLECTED= [ 3rd_grid  4th_grid  1st_grid  2nd_grid  7th_grid  8th_grid  5th_grid  6th_grid ] (MATLAB)
        #REFLECTED= [ 2nd lvl   3rd lvl   0th lvl   1st lvl   6th lvl   7th lvl   4th lvl   5th lvl  ] (PYTHON) 
        
        #BOUNCE BACK DENSITIES for next time-step
        f[2,ON_i,ON_j] = BOUNCEDBACK[:,0]
        f[3,ON_i,ON_j] = BOUNCEDBACK[:,1]
        f[0,ON_i,ON_j] = BOUNCEDBACK[:,2]
        f[1,ON_i,ON_j] = BOUNCEDBACK[:,3]
        f[6,ON_i,ON_j] = BOUNCEDBACK[:,4]
        f[7,ON_i,ON_j] = BOUNCEDBACK[:,5]
        f[4,ON_i,ON_j] = BOUNCEDBACK[:,6]
        f[5,ON_i,ON_j] = BOUNCEDBACK[:,7]


        #Updates simulation parameters
        ts=ts+1   # update time step
        
        # Save files info!
        ctsave = ctsave + 1
        if (ctsave % print_dump) == 0:

            # increment pSave counter
            pSave = pSave + 1

            # compute vorticity
            dUx_y = ( UX[0:Nx-1,1:Ny] - UX[0:Nx-1,0:Ny-1] ) / dy
            dUy_x = ( UY[1:Nx,0:Ny-1] - UY[0:Nx-1,0:Ny-1] ) / dx
            vorticity[0:Nx-1,0:Ny-1] = dUy_x - dUx_y 

            # print to vtk
            print_vtk_files(pSave,UX,UY,vorticity,Lx,Ly,Nx,Ny,Bound_Pts)
            print('Simulation Time: {0:6.6f}\n'.format(ts))
        

#### ENDS MAIN TIME-STEPPING ROUTINE #####


if __name__ == "__main__":
    lets_do_LBM()    
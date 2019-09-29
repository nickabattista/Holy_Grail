'''
 Solves the Navier-Stokes equations in the Velocity-Pressure formulation 
 using a predictor-corrector projection method approach

 Author: Nicholas A. Battista
 Created: Novermber 24, 2014 (MATLAB)
 Modified: December 8, 2014  (MATLAB)
 Created: April 25, 2017 (Python3)
 Modified: September 11, 2019 (Python3)

 Equations of Motion:
 Du/Dt = -Nabla(P) + nu*Laplacian(u)  [Conservation of Momentum]
 Nabla \cdot u = 0                    [Conservation of Mass]                                   


 IDEA: for each time-step
       1. Compute an intermediate velocity field explicitly, 
          use the momentum equation, but ignore the pressure gradient term
       2. Solve the Poisson problem for the pressure, whilst enforcing
          that the true velocity is divergence free. 
       3. Projection Step: correct the intermediate velocity field to
          obtain a velocity field that satisfies momentum and
          incompressiblity.
          
  '''

import numpy as np
from scipy import misc, fftpack
import os
from print_vtk_files import *
import time

###########################################################################

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

###########################################################################
#
# Function that chooses which simulation to run by changing the BCs
#
###########################################################################

def print_Projection_Info():

    print('\n____________________________________________________________________________\n')
    print('\nSolves the Navier-Stokes equations in the Velocity-Pressure formulation \n')
    print('using a predictor-corrector projection method approach\n\n')
    print('Author: Nicholas A. Battista\n')
    print('Created: Novermber 24, 2014\n')
    print('Modified: September 11, 2019\n')
    print('____________________________________________________________________________\n\n')
    print('Equations of Motion:\n')
    print('Du/Dt = -Nabla(u) + nu*Laplacian(u)  [Conservation of Momentum] \n')
    print('Nabla cdot u = 0                     [Conservation of Mass -> Incompressibility]     \n\n')                              
    print('IDEA: for each time-step\n')
    print('       1. Compute an intermediate velocity field explicitly, \n')
    print('          use the momentum equation, but ignore the pressure gradient term\n')
    print('       2. Solve the Poisson problem for the pressure, whilst enforcing\n')
    print('          that the true velocity is divergence free. \n')
    print('       3. Projection Step: correct the intermediate velocity field to\n')
    print('          obtain a velocity field that satisfies momentum and\n')
    print('          incompressiblity.\n\n')
    print('____________________________________________________________________________\n\n')



##############################################################################
#
# Function to initialize storage for velocities, pressure, vorticity,
# coeffs., etc.
#
##############################################################################

def initialize_Storage(Nx,Ny):

    #Initialize (u,v) velocities, pressure
    u=np.zeros((Nx+1,Ny+2))      # x-velocity (u) initially zero on grid
    v=np.zeros((Nx+2,Ny+1))      # y-velocity (v) initially zero on grid
    p=np.zeros((Nx+2,Ny+2))      # pressure initially zero on grid
    uTemp=np.zeros((Nx+1,Ny+2))  # auxillary x-Velocity (u*) field initially zero on grid
    vTemp=np.zeros((Nx+2,Ny+1))  # auxillary y-Velocity (v*) field initially zero on grid

    #Initialize quantities for plotting
    vorticity=np.zeros((Nx+1,Ny+1)) # w: Vorticity

    #Coefficients when solving the Elliptic Pressure Equation w/ SOR (so averaging is consistent)
    c=1/4*np.ones((Nx+2,Ny+2))  # Interior node coefficients set to 1/4 (all elements exist)
    c[1,2:Ny-1]=1/3             # Boundary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c[Nx,2:Ny-1]=1/3            # Boundary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c[2:Nx-1,1]=1/3             # Boundary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c[2:Nx-1,Ny]=1/3            # Boundary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c[1,1]=1/2                  # Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c[1,Ny]=1/2                 # Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c[Nx,1]=1/2                 # Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c[Nx,Ny]=1/2                # Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    
    return u, v, p, uTemp, vTemp, vorticity, c


###########################################################################
#
# Function that chooses which simulation to run by changing the BCs
# 
#       NOTE: all velocities normal to the boundary are zero. 
#
###########################################################################
    
def please_Give_Me_BCs(choice):

# Possible choices: 'cavity_top', 'whirlwind', 'twoSide_same', 'twoSide_opp', 'corner'

    if (choice == 'cavity_top'):

        bVel = 4.0
        uTop = bVel 
        uBot = 0.0 
        vRight = 0.0 
        vLeft = 0.0

        endTime = 6.0                # Final Simulation Time(s)
        dt = 0.001                   # Time-step
        nStep=np.floor(endTime/dt)   # Number of Time-Steps

    elif (choice=='whirlwind'):

        bVel = 1.0
        uTop=bVel  
        uBot=-bVel 
        vRight=-bVel 
        vLeft=bVel

        endTime = 24                 # Final Simulation Time(s)
        dt = 0.001                   # Time-step
        nStep=np.floor(endTime/dt)   # Number of Time-Steps


    elif (choice=='twoSide_same'):

        bVel = 2.0
        uTop = 0.0 
        uBot = 0.0 
        vRight = bVel 
        vLeft = bVel

        dt = 0.01      #Time-step
        nStep=300      #Number of Time-Steps


    elif (choice=='twoSide_opp'):

        bVel = 2.0
        uTop = 0.0 
        uBot = 0.0 
        vRight = -bVel 
        vLeft = bVel

        dt = 0.01      #Time-step
        nStep=300      #Number of Time-Steps


    elif (choice=='corner'):

        bVel = 1.0
        uTop = bVel 
        uBot = 0.0 
        vRight = 0 
        vLeft = bVel

        dt = 0.01      #Time-step
        nStep=300      #Number of Time-Steps


    else:

        print('YOU DID NOT CHOOSE CORRECTLY!!!!!\n')
        print('Simulation DEFAULT: whirlwind\n')

        bVel = 1.0
        uTop=bVel  
        uBot=-bVel 
        vRight=-bVel 
        vLeft=bVel

        endTime = 24                 # Final Simulation Time(s)
        dt = 0.001                   # Time-step
        nStep=np.floor(endTime/dt)   # Number of Time-Steps


    return uTop,uBot,vRight,vLeft,dt,nStep,bVel 



###########################################################################
#
# Function that chooses which simulation to run by changing the BCs
#
###########################################################################

def print_Simulation_Info(choice,dt,dx,nu,bVel,Lx,Ly):

    # PRINT STABILITY INFO #
    print('\nNOTE: dt must be <= {0:6.6f} for any chance of STABILITY!\n'.format(0.25*dx*dx/nu))
    print('Your dt = {0:6.6f}\n\n'.format(dt))

    if (choice=='cavity_top'):

        print('You are simulating cavity flow\n')
        print('The open cavity is on the TOP side\n')
        print('Try changing the viscosity or geometry\n\n')

    elif (choice=='whirlwind'):

        print('You are simulating vortical flow\n')
        print('All velocities on the wall point at the next corner in a CW manner\n')
        print('Try changing the velocity BCs on each wall')
        print('Or try changing the viscosity or geometry\n\n')

    elif (choice=='twoSide_same'):

        print('You are simulating two-sided cavity flow\n')
        print('The open cavities are on the left and right sides\n')
        print('The vertical velocities n those walls point in the same direction\n')
        print('Try changing the viscosity or geometry\n\n')

    elif (choice=='twoSide_opp'):

        print('You are simulating two-sided cavity flow\n')
        print('The open cavities are on the left and right sides\n')
        print('The vertical velocities on those walls point in opposite directions\n')
        print('Try changing the viscosity or geometry\n\n')

    elif (choice=='corner'):

        print('You are simulating cavity flow, through the left and top boundaries\n')
        print('The vertical velocities on those walls point in opposite directions\n')
        print('Try changing the viscosity or geometry\n\n')

    else:

        print('YOU DID NOT CHOOSE CORRECTLY.\n')
        print('Simulation default: whirlwind\n')

    #Prints Re Number # 
    print('\nYour Re is: {0:6.6f}\n\n'.format(Lx*bVel/nu))
    print('____________________________________________________________________________\n\n')


##############################################################################
#
# Function to find auxillary (temporary) velocity fields in predictor step
#
##############################################################################

def give_Auxillary_Velocity_Fields(dt,dx,nu,Nx,Ny,u,v,uTemp,vTemp):
    

    #Find Temporary u-Velocity Field
    for ii in range(2,Nx+1): 
        i=ii-1
        for jj in range(2,Ny+2):
            j=jj-1
            uTemp[i,j]=u[i,j]+dt*(-(0.25/dx)*((u[i+1,j]+u[i,j])**2-(u[i,j]+u[i-1,j])**2+(u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j])-(u[i,j]+u[i,j-1])*(v[i+1,j-1]+v[i,j-1]))+(nu/dx**2)*(u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]-4*u[i,j]) )

    #Find Temporary v-Velocity Field
    for ii in range(2,Nx+2):
        i=ii-1
        for jj in range(2,Ny+1): 
            j=jj-1
            vTemp[i,j]=v[i,j]+dt*(-(0.25/dx)*((u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j])-(u[i-1,j+1]+u[i-1,j])*(v[i,j]+v[i-1,j])+(v[i,j+1]+v[i,j])**2-(v[i,j]+v[i,j-1])**2)+(nu/dx**2)*(v[i+1,j]+v[i-1,j]+v[i,j+1]+v[i,j-1]-4*v[i,j]))

    return uTemp, vTemp


##############################################################################
#
# Function to solve elliptic pressure equation using a SOR scheme 
#
##############################################################################

def solve_Elliptic_Pressure_Equation(dt,dx,Nx,Ny,maxIter,beta,c,uTemp,vTemp,p):

    iter = 1 
    err = 1 
    tol = 5e-6
    
    pPrev = np.array(p)
    while ( (err > tol) and (iter < maxIter) ):    
        for ii in range(2,Nx+2): 
            i=ii-1
            for jj in range(2,Ny+2):
                j=jj-1
                p[i,j]=beta*c[i,j]*(p[i+1,j]+p[i-1,j]+p[i,j+1]+p[i,j-1]-(dx/dt)*(uTemp[i,j]-uTemp[i-1,j]+vTemp[i,j]-vTemp[i,j-1]))+(1-beta)*p[i,j]

        err = np.max( abs( p - pPrev ) ) 

        pPrev = np.array(p)
        print(err)
        iter = iter + 1
        
    return p


###########################################################################
#
# Function that performs the PROJECTION METHOD SIMULATION!
#
###########################################################################

def Projection_Method():

#
# Solves the Navier-Stokes equations in the Velocity-Pressure formulation 
# using a predictor-corrector projection method approach
#
# Author: Nicholas A. Battista
# Created: Novermber 24, 2014 (MATLAB)
# Modified: December 8, 2014  (MATLAB)
# Created: April 25, 2017     (Python3)
# Modified: September 11, 2019 (Python3) 
#
# Equations of Motion:
# Du/Dt = -Nabla(P) + nu*Laplacian(u)  [Conservation of Momentum]
# Nabla \cdot u = 0                    [Conservation of Mass]                                   
#
#
# IDEA: for each time-step
#       1. Compute an intermediate velocity field explicitly, 
#          use the momentum equation, but ignore the pressure gradient term
#       2. Solve the Poisson problem for the pressure, whilst enforcing
#          that the true velocity is divergence free. 
#       3. Projection Step: correct the intermediate velocity field to
#          obtain a velocity field that satisfies momentum and
#          incompressiblity.
#

    # Print key fluid solver ideas to screen
    print_Projection_Info()

    #
    # GRID PARAMETERS #
    #
    Lx = 1.0        # Domain Length in x
    Ly = 2.0        # Domain Length in y
    Nx = 128        # Spatial Resolution in x
    Ny = 256        # Spatial Resolution in y
    dx=Lx/Nx        # Spatial Distance Definition in x (NOTE: keep dx = Lx/Nx = Ly/Ny = dy)


    #
    # SIMULATION PARAMETERS #
    #
    mu = 1.0        # Fluid DYNAMIC viscosity (kg / m*s) 
    #                (mu=1000.0,100.0,10.0,1.0 for Re=4,40,400,4000 respectively for Cavity Flow examples
    #                (mu=0.25,1.0,2.5 for Re=4000,1000, and 400, respectively for Circular Flow examples)
    rho = 1000.0    # Fluid DENSITY (kg/m^3) 
    nu=mu/rho       # Fluid KINEMATIC viscosity
    numPredCorr = 3 # Number of Predictor-Corrector Steps
    maxIter=200     # Maximum Iterations for SOR Method to solve Elliptic Pressure Equation
    beta=1.25       # Relaxation Parameter 


    #
    # CHOOSE SIMULATION (gives chosen simulation parameters) #
    # Possible choices: 'cavity_top', 'whirlwind', 'twoSide_same', 'twoSide_opp', 'corner'
    #
    choice = 'cavity_top'
    uTop,uBot,vRight,vLeft,dt,nStep,bVel = please_Give_Me_BCs(choice)


    #
    # Initialize all storage quantities #
    #
    u, v, p, uTemp, vTemp, vorticity, c = initialize_Storage(Nx,Ny)

    


    #PRINT SIMULATION INFO #
    print_Simulation_Info(choice,dt,dx,nu,bVel,Lx,Ly) 


    # SAVING DATA TO VTK #
    print_dump = 200 # Saves data every print_dump time-steps
    ctsave = 0       # Keeps track of total time-steps
    pCount = 0       # Counter for # of time-steps saved for indexing data

    
    # CREATE VTK DATA FOLDER #
    try:
        os.mkdir('vtk_data')
    except FileExistsError:
        #Folder already exists
        pass
    
    # PRINT INITIAL DATA TO VTK #
    print_vtk_files(pCount,u,v,p,vorticity,Lx,Ly,Nx,Ny)

    # Initial time for simulation
    t=0.0 #Initialize time
    print('Simulation Time: {0:6.6f}\n'.format(t))

    #
    # BEGIN TIME-STEPIN'!
    # 
    for j in range(1,int(nStep)+1):

        #Enforce Boundary Conditions (Solve for "ghost velocities")
        u[0:Nx,0]   = ( 2*uBot-u[0:Nx,1]  )  * np.tanh(0.25*t)
        u[0:Nx,Ny+1]= ( 2*uTop-u[0:Nx,Ny] )  * np.tanh(0.25*t)
        v[0,0:Ny]   = ( 2*vLeft-v[1,0:Ny] )  * np.tanh(0.25*t)
        v[Nx+1,0:Ny]= ( 2*vRight-v[Nx,0:Ny] )* np.tanh(0.25*t)

        print(ctsave)

        #Start Predictor-Corrector Steps
        for k in range(1,numPredCorr+1):
            
            #Find auxillary (temporary) velocity fields for predictor step
            uTemp, vTemp = give_Auxillary_Velocity_Fields(dt,dx,nu,Nx,Ny,u,v,uTemp,vTemp)

            #Solve Elliptic Equation for Pressure via SOR scheme
            p = solve_Elliptic_Pressure_Equation(dt,dx,Nx,Ny,maxIter,beta,c,uTemp,vTemp,p)

            # Velocity Correction
            u[1:Nx-1,1:Ny] = uTemp[1:Nx-1,1:Ny] - (dt/dx) * ( p[2:Nx,1:Ny] - p[1:Nx-1,1:Ny] )
            v[1:Nx,1:Ny-1] = vTemp[1:Nx,1:Ny-1] - (dt/dx) * ( p[1:Nx,2:Ny] - p[1:Nx,1:Ny-1] )


        #Update Simulation Time 
        t=t+dt

        # Save files info!
        ctsave = ctsave + 1

        if ( ctsave % print_dump == 0):

            # increment data storage counter
            pCount = pCount + 1

            # compute vorticity
            vorticity[0:Nx,0:Ny] = ( u[0:Nx,1:Ny+1] - u[0:Nx,0:Ny] - v[1:Nx+1,0:Ny] + v[0:Nx,0:Ny] ) / (2*dx) 

            # call function to store data at this time-step
            print_vtk_files(pCount,u,v,p,vorticity,Lx,Ly,Nx,Ny)
            
            # print simulation time
            print('Simulation Time: {0:6.6f}\n'.format(t))

        #
        #ENDS PROJECTION METHOD TIME-STEPPING
        #

    ##############################################################################


if __name__ == "__main__":
    Projection_Method()
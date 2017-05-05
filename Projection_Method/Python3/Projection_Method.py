'''
 Solves the Navier-Stokes equations in the Velocity-Pressure formulation 
 using a predictor-corrector projection method approach

 Author: Nicholas A. Battista
 Created: Novermber 24, 2014 (MATLAB)
 Modified: December 8, 2014  (MATLAB)
 Created: April 25, 2017 (Python 3)
 
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
    print('Modified: December 8, 2014\n')
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

def initialize_Storage(nx,ny):

    #Initialize (u,v) velocities, pressure
    u=np.zeros((nx+1,ny+2))      # x-velocity (u) initially zero on grid
    v=np.zeros((nx+2,ny+1))      # y-velocity (v) initially zero on grid
    p=np.zeros((nx+2,ny+2))      # pressure initially zero on grid
    uTemp=np.zeros((nx+1,ny+2))  # auxillary x-Velocity (u*) field initially zero on grid
    vTemp=np.zeros((nx+2,ny+1))  # auxillary y-Velocity (v*) field initially zero on grid

    #Initialize quantities for plotting
    uAvg=np.zeros((nx+1,ny+1))      # uAvg: averaged x-Velocities on grid (smoothing)
    vAvg=np.zeros((nx+1,ny+1))      # vAvg: averaged y-Velocities on grid (smoothing)
    vorticity=np.zeros((nx+1,ny+1)) # w: Vorticity

    #Coefficients when solving the Elliptic Pressure Equation w/ SOR (so averaging is consistent)
    c=1/4*np.ones((nx+2,ny+2))  # Interior node coefficients set to 1/4 (all elements exist)
    c[1,2:ny-1]=1/3             # BouTopdary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c[nx,2:ny-1]=1/3            # BouTopdary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c[2:nx-1,1]=1/3             # BouTopdary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c[2:nx-1,ny]=1/3            # BouTopdary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c[1,1]=1/2                  # Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c[1,ny]=1/2                 # Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c[nx,1]=1/2                 # Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c[nx,ny]=1/2                # Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    
    return u, v, p, uTemp, vTemp, uAvg, vAvg, vorticity, c


###########################################################################
#
# Function that chooses which simulation to run by changing the BCs
#
###########################################################################
    
def please_Give_Me_BCs(choice):

# Possible choices: 'cavity_left', 'whirlwind', 'twoSide_same', 'twoSide_opp', 'corner'

    if (choice == 'cavity_left'):

        bVel = 4.0
        uTop = 0.0 
        uBot = 0.0 
        vRight = 0.0 
        vLeft = bVel

        endTime = 6.0
        dt = 0.00125                 # Time-step
        nStep=np.floor(endTime/dt)   # Number of Time-Steps
        printStep = 5                # Print ever y# of printStep frames

    elif (choice=='whirlwind'):

        bVel = 1.0
        uTop=bVel  
        uBot=-bVel 
        vRight=-bVel 
        vLeft=bVel

        dt = 0.01      #Time-step
        nStep=150      #Number of Time-Steps
        printStep = 2  #Print every # of printStep frames


    elif (choice=='twoSide_same'):

        bVel = 2.0
        uTop = 0.0 
        uBot = 0.0 
        vRight = bVel 
        vLeft = bVel

        dt = 0.01      #Time-step
        nStep=300      #Number of Time-Steps
        printStep = 3  #Print every # of printStep frames


    elif (choice=='twoSide_opp'):

        bVel = 2.0
        uTop = 0.0 
        uBot = 0.0 
        vRight = -bVel 
        vLeft = bVel

        dt = 0.01      #Time-step
        nStep=300      #Number of Time-Steps
        printStep = 3  #Print every # of printStep frames


    elif (choice=='corner'):

        bVel = 1.0
        uTop = bVel 
        uBot = 0.0 
        vRight = 0 
        vLeft = bVel

        dt = 0.01      #Time-step
        nStep=300      #Number of Time-Steps
        printStep = 3  #Print every # of printStep frames


    else:

        print('YOU DID NOT CHOOSE CORRECTLY!!!!!\n')
        print('Simulation DEFAULT: whirlwind\n')

        bVel = 1.0
        uTop=bVel  
        uBot=-bVel 
        vRight=-bVel 
        vLeft=bVel

        dt = 0.01      #Time-step
        nStep=150      #Number of Time-Steps
        printStep = 2  #Print every # of printStep frames


    return uTop,uBot,vRight,vLeft,dt,nStep,printStep,bVel #,xStart,yStart



###########################################################################
#
# Function that chooses which simulation to run by changing the BCs
#
###########################################################################

def print_Simulation_Info(choice,dt,dx,nu,bVel,Ly):

    # PRINT STABILITY INFO #
    print('\nNOTE: dt must be <= {0:6.6f} for STABILITY!\n'.format(0.25*dx*dx/nu))
    print('Your dt = {0:6.6f}\n\n'.format(dt))

    if (choice=='cavity_left'):

        print('You are simulating cavity flow\n')
        print('The open cavity is on the left side\n')
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
    print('\nYour Re is: {0:6.6f}\n\n'.format(Ly*bVel/nu))
    print('____________________________________________________________________________\n\n')


##############################################################################
#
# Function to find auxillary (temporary) velocity fields in predictor step
#
##############################################################################

def give_Auxillary_Velocity_Fields(dt,dx,nu,nx,ny,u,v,uTemp,vTemp):
    

    #Find Temporary u-Velocity Field
    for ii in range(2,nx+1): 
        i=ii-1
        for jj in range(2,ny+2):
            j=jj-1
            uTemp[i,j]=u[i,j]+dt*(-(0.25/dx)*((u[i+1,j]+u[i,j])**2-(u[i,j]+u[i-1,j])**2+(u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j])-(u[i,j]+u[i,j-1])*(v[i+1,j-1]+v[i,j-1]))+(nu/dx**2)*(u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]-4*u[i,j]))

    #Find Temporary v-Velocity Field
    for ii in range(2,nx+2):
        i=ii-1
        for jj in range(2,ny+1): 
            j=jj-1
            vTemp[i,j]=v[i,j]+dt*(-(0.25/dx)*((u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j])-(u[i-1,j+1]+u[i-1,j])*(v[i,j]+v[i-1,j])+(v[i,j+1]+v[i,j])**2-(v[i,j]+v[i,j-1])**2)+(nu/dx**2)*(v[i+1,j]+v[i-1,j]+v[i,j+1]+v[i,j-1]-4*v[i,j]))
    
    return uTemp, vTemp


##############################################################################
#
# Function to solve elliptic pressure equation using a SOR scheme 
#
##############################################################################

def solve_Elliptic_Pressure_Equation(dt,dx,nx,ny,maxIter,beta,c,uTemp,vTemp,p):

    iter = 1 
    err = 1 
    tol = 5e-6
    pPrev = p
    while ( (err > tol) and (iter < maxIter) ):    
        for ii in range(2,nx+2): 
            i=ii-1
            for jj in range(2,ny+2):
                j=jj-1
                p[i,j]=beta*c[i,j]*(p[i+1,j]+p[i-1,j]+p[i,j+1]+p[i,j-1]-(dx/dt)*(uTemp[i,j]-uTemp[i-1,j]+vTemp[i,j]-vTemp[i,j-1]))+(1-beta)*p[i,j]

        err = np.max( abs( p - pPrev ) ) 
        pPrev = p
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
    Lx = 2.0                      # Lenght in x
    Ly = 1.0                      # Length in y
    nx=256                        # Initialize X-Grid (Spatial Resolution in x)
    ny=128                        # Initialize Y-Grid (Spatial Resolution in y)
    dx=Lx/nx                      # Spatial Distance Definition in x
    xGrid = np.arange(0,Lx+dx,dx) # xGrid
    yGrid = np.arange(0,Ly+dx,dx) # yGrid


    #
    # SIMULATION PARAMETERS #
    #
    mu = 10         # Fluid DYNAMIC viscosity (kg / m*s)
    rho = 1000      # Fluid density (kg/m^3) 
    nu=mu/rho       # Fluid KINEMATIC viscosity
    numPredCorr = 3 # Number of Predictor-Corrector Steps
    maxIter=200     # Maximum Iterations for SOR Method to solve Elliptic Pressure Equation
    beta=1.25       # Relaxation Parameter 


    #
    # Initialize all storage quantities #
    #
    u, v, p, uTemp, vTemp, uAvg, vAvg, vorticity, c = initialize_Storage(nx,ny)

    
    #
    # CHOOSE SIMULATION (gives chosen simulation parameters) #
    # Possible choices: 'cavity_left', 'whirlwind', 'twoSide_same', 'twoSide_opp', 'corner'
    #
    choice = 'cavity_left'
    uTop,uBot,vRight,vLeft,dt,nStep,pStep,bVel = please_Give_Me_BCs(choice)


    #PRINT SIMULATION INFO #
    print_Simulation_Info(choice,dt,dx,nu,bVel,Ly) 


    # SAVING DATA TO VTK #
    print_dump = 40
    ctsave = 0

    
    # CREATE VTK DATA FOLDER #
    try:
        os.mkdir('vtk_data')
    except FileExistsError:
        #Folder already exists
        pass
    
    # PRINT INITIAL DATA TO VTK #
    print_vtk_files(ctsave,u,v,p,vorticity,Lx,Ly,nx,ny)

    # Initial time for simulation
    t=0.0 #Initialize time
    print('Simulation Time: {0:6.6f}\n'.format(t))

    #
    # BEGIN TIME-STEPIN'!
    # 
    for j in range(1,int(nStep)+1):

        #Enforce Boundary Conditions (Solve for "ghost velocities")
        u[0:nx,0]   = ( 2*uBot-u[0:nx,1]  )  * np.tanh(2.5*t)
        u[0:nx,ny+1]= ( 2*uTop-u[0:nx,ny] )  * np.tanh(2.5*t)
        v[0,0:ny]   = ( 2*vLeft-v[1,0:ny] )  * np.tanh(2.5*t)
        v[nx+1,0:ny]= ( 2*vRight-v[nx,0:ny] )* np.tanh(2.5*t)

        #Start Predictor-Corrector Steps
        for k in range(1,numPredCorr+1):

            #Find auxillary (temporary) velocity fields for predictor step
            uTemp, vTemp = give_Auxillary_Velocity_Fields(dt,dx,nu,nx,ny,u,v,uTemp,vTemp)

            #Solve Elliptic Equation for Pressure via SOR scheme
            p = solve_Elliptic_Pressure_Equation(dt,dx,nx,ny,maxIter,beta,c,uTemp,vTemp,p)

            # Velocity Correction
            u[1:nx-1,1:ny] = uTemp[1:nx-1,1:ny] - (dt/dx) * ( p[2:nx,1:ny] - p[1:nx-1,1:ny] )
            v[1:nx,1:ny-1] = vTemp[1:nx,1:ny-1] - (dt/dx) * ( p[1:nx,2:ny] - p[1:nx,1:ny-1] )
        
        #Update Simulation Time (not needed in algorithm)
        t=t+dt


        # Save files info!
        ctsave = ctsave + 1
        if ( ctsave % print_dump == 0):
            vorticity[0:nx,0:ny] = ( u[0:nx,1:ny+1] - u[0:nx,0:ny] - v[1:nx+1,0:ny] + v[0:nx,0:ny] ) / (2*dx) 
            print_vtk_files(ctsave,u,v,p,vorticity,Lx,Ly,nx,ny)
            print('Simulation Time: {0:6.6f}\n'.format(t))

        #
        #ENDS PROJECTION METHOD TIME-STEPPING
        #

    ##############################################################################



if __name__ == "__main__":
    Projection_Method()
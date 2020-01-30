'''Solves the Navier-Stokes equations in the Vorticity-Stream Function
 formulation using a pseudo-spectral approach w/ FFT

 Author: Nicholas A. Battista
 Created: Novermber 29, 2014 (MATLAB VERSION)
 Created: April 27, 2017 (PYTHON3 VERSION)
 MODIFIED: January 30, 2020 (PYTHON3 VERSION)
 
 Equations of Motion:
 D (Vorticity) /Dt = nu*Laplacian(Vorticity)  
 Laplacian(Psi) = - Vorticity                                                       

      Real Space Variables                   Fourier (Frequency) Space                                                          
       SteamFunction: Psi                     StreamFunction: Psi_hat
 Velocity: u = Psi_y & v = -Psi_x              Velocity: u_hat ,v_hat
         Vorticity: Vort                        Vorticity: Vort_hat


 IDEA: for each time-step
       1. Solve Poisson problem for Psi (in Fourier space)
       2. Compute nonlinear advection term by finding u and v in real
          variables by doing an inverse-FFT, compute advection, transform
          back to Fourier space
       3. Time-step vort_hat in frequency space using a semi-implicit
          Crank-Nicholson scheme (explicit for nonlinear adv. term, implicit
          for viscous term)'''

import numpy as np
from scipy import misc, fftpack
import os
import sys
from print_vtk_files import *


# Path Reference to where Initial Velocity Vector Fields are found #
sys.path.append('Get_Initial_Vorticity_From_Stored_Vector_Field/')
from give_Initial_Vorticity_From_Velocity_Field_VTK_File import *

###########################################################################
#
# Function to print information about fluid solver
#
###########################################################################

def print_FFT_NS_Info():

    print('\n_________________________________________________________________________\n\n')
    print(' \nSolves the Navier-Stokes equations in the Vorticity-Stream Function \n')
    print(' formulation using a pseudo-spectral approach w/ FFT \n\n')
    print(' Author: Nicholas A. Battista \n')
    print(' Created: Novermber 29, 2014 \n')
    print(' Modified: January 30, 2020 \n\n')
    print(' Equations of Motion: \n')
    print(' D (Vorticity) /Dt = nu*Laplacian(Vorticity)  \n')
    print(' Laplacian(Psi) = - Vorticity                 \n\n')                                     
    print('      Real Space Variables                   Fourier (Frequency) Space              \n')                                            
    print('       SteamFunction: Psi                     StreamFunction: Psi_hat \n')
    print(' Velocity: u = Psi_y & v = -Psi_x              Velocity: u_hat ,v_hat \n')
    print('         Vorticity: Vort                        Vorticity: Vort_hat \n\n')
    print('_________________________________________________________________________\n\n')
    print(' IDEA: for each time-step \n')
    print('       1. Solve Poisson problem for Psi (in Fourier space)\n')
    print('       2. Compute nonlinear advection term by finding u and v in real \n')
    print('          variables by doing an inverse-FFT, compute advection, transform \n')
    print('          back to Fourier space \n')
    print('       3. Time-step vort_hat in frequency space using a semi-implicit \n')
    print('          Crank-Nicholson scheme (explicit for nonlinear adv. term, implicit \n')
    print('          for viscous term) \n')
    print('_________________________________________________________________________\n\n')




###########################################################################
#
# Function to print information about the specific selected simulation
#
###########################################################################

def print_Simulation_Info(choice):

    if (choice == 'bubble1' ):

        print('You are simulating one dense region of CW vorticity in a bed of random vorticity values\n')
        print('Try changing the kinematic viscosity to see how flow changes\n')
        print('_________________________________________________________________________\n\n')


    elif ( choice == 'bubble3' ):

        print('You are simulating three nested regions of Vorticity (CW,CCW,CW) in a bed of random vorticity values\n')
        print('Try changing the position of the nested vortices in the "please_Give_Initial_State" function\n')
        print('Try changing the kinematic viscosity to see how flow changes\n')
        print('_________________________________________________________________________\n\n')


    elif ( choice == 'bubbleSplit' ):

        print('You are simulating two vortices which are very close\n')
        print('Try changing the initial vorticity distribution on the left or right side\n')
        print('Try changing the kinematic viscosity to see how the flow changes\n')
        print('_________________________________________________________________________\n\n')


    elif ( choice == 'qtrs' ):

        print('You are simulating 4 squares of differing vorticity\n')
        print('Try changing the initial vorticity in each square to see how the dynamics change\n')
        print('Try changing the kinematic viscosity to see how the flow changes\n')
        print('_________________________________________________________________________\n\n')


    elif (choice == 'half' ):

        print('You are simulating two half planes w/ opposite sign vorticity\n')
        print('Try changing the initial vorticity on each side to see how the dynamics change\n')
        print('Try changing the kinematic viscosity to see how the flow changes\n')
        print('_________________________________________________________________________\n\n')


    elif (choice == 'rand'):

        print('You are simulating a field of random vorticity values\n')
        print('Try changing the kinematic viscosity to see how the flow changes\n')
        print('_________________________________________________________________________\n\n')

    elif (choice == 'jets'):

        print('You are simulating the evolution of vorticity from an initial velocity field\n')
        print('Try changing the kinematic viscosity to see how the flow changes\n')
        print('Try importing a different initial velocity field\n')
        print('_________________________________________________________________________\n\n')


###########################################################################
#
# Function to choose initial vorticity state
#
###########################################################################

def please_Give_Initial_Vorticity_State(choice,NX,NY):

    # choice: string for selecting which example
    # NX,NY: grid resolution in x,y repsectively

    if ( choice == 'half' ):

        #
        # USE RECTANGLE: Lx = 2Ly, Nx = 2Ny
        #

        # radii of vortex regions (given in terms of mesh widths)
        radius1 = 0.3*NY
        radius2 = 0.15*NY
        
        # repmat(a, m, n) is np.tile(a, (m, n)).
        # np.tile(M,(m,n))
        #a1=repmat(-(NX-1)/2:(NX-1)/2,[NY,1]);  (MATLAB)
        #a2=repmat(-(NY-1)/2:(NY-1)/2,[NX,1]);  (MATLAB)

        # auxillary vectors 
        xAux = np.arange(-(NX-1)/2, (NX-1)/2+1, 1)
        yAux = np.arange(-(NY-1)/2, (NY-1)/2+1, 1)
        
        # stack vectors to create grids of indices
        a1=np.tile(xAux,(NY,1))
        a2=np.tile(yAux,(NX,1))          

        # Amount to translate cylinder from middle of domain
        aR = np.floor(0.25*NX)    

        # Form circular regions of vorticity
        b1 = ( (a1+aR)**2+((a2).T)**2) < radius1**2
        b2 = ( (a1-aR)**2+((a2).T)**2) < radius2**2 

        # Convert to 0,1 boolean matrix from False,True matrix
        b1 = b1.astype(int)
        b2 = b2.astype(int)

        # Convert boolean matrix to matrix of double values
        # Note: here assuming no overlapping vorticity regions
        b1 = np.double(b1) + np.double(b2)

        # Find values where vorticity is
        [r1,c1]=np.nonzero(b1)
    
        vort = np.zeros((NX,NY))
        for i in range(0,r1.size):
            if c1[i] >= NX/2:
                vort[c1[i],r1[i]]= -0.05
            else:
                vort[c1[i],r1[i]]= -0.1 

        dt=1e-2        # time step
        tFinal = 5     # final time
        plot_dump=20   # interval for plots

    elif ( choice == 'qtrs' ):


        #
        # USE SQUARE: Lx = Ly, Nx = Ny
        #
    
        # radii of vortex regions (given in terms of mesh widths)
        radius11 = 0.2*NY
        radius12 = 0.2*NY
        radius21 = 0.2*NY
        radius22 = 0.2*NY

        # auxillary vectors 
        xAux = np.arange(-(NX-1)/2, (NX-1)/2+1, 1)
        yAux = np.arange(-(NY-1)/2, (NY-1)/2+1, 1)

        # stack vectors to create grids of indices
        a1=np.tile(xAux,(NY,1))
        a2=np.tile(yAux,(NX,1))          

        # Amount to translate cylinder from middle of domain
        aR = np.floor(0.25*NX)  
        aU = np.floor(0.25*NY)      

        # Form circular regions of vorticity
        b1 = ( (a1+aR)**2+((a2+aU).T)**2) < radius11**2
        b2 = ( (a1-aR)**2+((a2-aU).T)**2) < radius12**2 
        b3 = ( (a1+aR)**2+((a2-aU).T)**2) < radius21**2
        b4 = ( (a1-aR)**2+((a2+aU).T)**2) < radius22**2 

        # Convert to 0,1 boolean matrix from False,True matrix
        b1 = b1.astype(int)
        b2 = b2.astype(int)
        b3 = b3.astype(int)
        b4 = b4.astype(int)

        # Convert boolean matrix of 0,1's to matrix of double values
        b1 = np.double(b1) + np.double(b2) + np.double(b3) + np.double(b4)

        # Find values where vorticity is
        [r1,c1]=np.nonzero(b1)
    
        vort = np.zeros((NX,NY))
        for i in range(0,r1.size):
            if c1[i] >= NX/2:
                if r1[i]>=NY/2:
                    vort[c1[i],r1[i]]= 0.1
                else:
                    vort[c1[i],r1[i]]= -0.1
            else:
                if r1[i]>=NY/2:
                    vort[c1[i],r1[i]]= 0.1
                else:
                    vort[c1[i],r1[i]]= -0.1 

        dt=1e-2        # time step
        tFinal = 5     # final time
        plot_dump=20   # interval for plots


    elif ( choice == 'rand' ):

        #
        # Any domain is fine, as long as Lx/Nx = Ly/Ny
        #

        vort = 2*np.random.rand(NX,NY)-1
        dt=1e-1       # time step
        tFinal = 1000 # final time
        plot_dump=25  # interval for plots

    elif ( choice == 'jets' ):
    
        #
        # DOMAIN MUST BE SQUARE AND 256x256 FOR THIS BUILT-IN EXAMPLE.
        #
        vort = give_Initial_Vorticity_From_Velocity_Field_VTK_File()
        vort = vort / 1000
        
        dt = 1e-2        # time step
        tFinal = 50      # final time
        plot_dump= 100   # interval for plots 
    
    elif ( choice == 'bubble1' ):
        
        #
        # USE RECTANGLE: Lx = Ly, Nx = Ny
        #

        # radii of vortex regions (given in terms of mesh widths)
        radius1 = 0.25*NX

        # auxillary vectors 
        xAux = np.arange(-(NX-1)/2, (NX-1)/2+1, 1)
        yAux = np.arange(-(NY-1)/2, (NY-1)/2+1, 1)

        # stack vectors to create grids of indices
        a1=np.tile(xAux,(NY,1))
        a2=np.tile(yAux,(NX,1))          

        # Form circular regions of vorticity
        b1 = ( (a1)**2+((a2).T)**2) < radius1**2         # region at center of domain

        # Convert to 0,1 boolean matrix from False,True matrix
        b1 = b1.astype(int)

        # Initialize vorticity in grid to random values between -1,1
        vort = 2*np.random.rand(NX,NY)-1

        # Find values where largest region is
        [r1,c1]=np.nonzero(b1)
        for i in range(0,r1.size):
            vort[c1[i],r1[i]] = 0.6   

        dt = 1e-2       # time step
        tFinal = 30     # final time
        plot_dump= 50   # interval for plots

    elif ( choice == 'bubbleSplit'):

        #
        # USE RECTANGLE: Lx = Ly, Nx = Ny
        #

        # radii of vortex regions (given in terms of mesh widths)
        radius1 = 0.25*NX

        # auxillary vectors 
        xAux = np.arange(-(NX-1)/2, (NX-1)/2+1, 1)
        yAux = np.arange(-(NY-1)/2, (NY-1)/2+1, 1)

        # stack vectors to create grids of indices
        a1=np.tile(xAux,(NY,1))
        a2=np.tile(yAux,(NX,1))          

        # Form circular regions of vorticity
        b1 = ( (a1)**2+((a2).T)**2) < radius1**2         # region at center of domain

        # Convert to 0,1 boolean matrix from False,True matrix
        b1 = b1.astype(int)

        # Initialize vorticity in grid to random values between -1,1
        vort = 2*np.random.rand(NX,NY)-1

        # Find values where largest region is
        [r1,c1]=np.nonzero(b1)
        for i in range(0,r1.size):
            if c1[i]<= NX/2:
                vort[c1[i],r1[i]] = -0.5*np.random.rand()   
            else:
                vort[c1[i],r1[i]] = 0.5*np.random.rand()  

        dt = 1e-2       # time step
        tFinal = 30     # final time
        plot_dump= 50   # interval for plots


    elif ( choice == 'bubble3' ):

        #
        # USE RECTANGLE: Lx = Ly, Nx = Ny
        #

        # radii of vortex regions (given in terms of mesh widths)
        radius1 = 0.25*NX
        radius2 = 0.175*NX
        radius3 = 0.10*NX

        # auxillary vectors 
        xAux = np.arange(-(NX-1)/2, (NX-1)/2+1, 1)
        yAux = np.arange(-(NY-1)/2, (NY-1)/2+1, 1)

        # stack vectors to create grids of indices
        a1=np.tile(xAux,(NY,1))
        a2=np.tile(yAux,(NX,1))          

        # Amount to translate cylinder from middle of domain
        aD = np.floor(0.10*NX)    
        aR = np.floor(0.15*NX)    

        # Form circular regions of vorticity
        b1 = ( (a1)**2+((a2).T)**2) < radius1**2         # region at center of domain
        b2 = ( (a1)**2+((a2-aD).T)**2) < radius2**2      # shift 2nd region down from center
        b3 = ( (a1+aR)**2+((a2-2*aD).T)**2) < radius3**2 # shift 3rd region down from center

        # Convert to 0,1 boolean matrix from False,True matrix
        b1 = b1.astype(int)
        b2 = b2.astype(int)
        b3 = b3.astype(int)

        # Initialize vorticity in grid to random values between -1,1
        vort = 2*np.random.rand(NX,NY)-1

        # Find values where largest region is
        [r1,c1]=np.nonzero(b1)
        for i in range(0,r1.size):
            vort[c1[i],r1[i]] = 0.4

        # Find values where 2ND largest region is
        [r2,c2]=np.nonzero(b2)
        for i in range(0,r2.size):
            vort[c2[i],r2[i]] = -0.5        

        # Find values where 3RD largest region is (smallest)
        [r3,c3]=np.nonzero(b3)
        for i in range(0,r3.size):
            vort[c3[i],r3[i]] = 0.5           

        dt = 1e-2       # time step
        tFinal = 30     # final time
        plot_dump= 50   # interval for plots
   
    ######### ENDS ALL EXAMPLE INITIALIZATIONS ##########

    # Finally transform initial vorticity state to frequency space using FFT
    vort_hat = fftpack.fft2(vort) 

    # Print simulation information
    print_Simulation_Info(choice)

    return vort_hat,dt,tFinal,plot_dump


###########################################################################
#
# FUNCTION: Initializes Wavenumber Matrices for FFT
#
###########################################################################

def please_Give_Wavenumber_Matrices(NX,NY):

    kMatx = np.zeros((NX,NY),dtype=np.complex_)
    kMaty = np.zeros((NX,NY),dtype=np.complex_)

    rowVec = np.concatenate( ( range(0,int(NY/2)+1) , -np.arange(int(NY/2)-1,0,-1) ), axis=0)
    colVec = np.concatenate( ( range(0,int(NX/2)+1) , -np.arange(int(NX/2)-1,0,-1) ), axis=0)

    #Makes wavenumber matrix in x
    for i in range(1,NX+1):
        kMatx[i-1,:] = 1j*rowVec  
    
    #Makes wavenumber matrix in y (NOTE: if Nx=Ny, kMatx = kMaty')
    for j in range(1,NY+1):
        kMaty[:,j-1] = 1j*colVec 
        
    # Laplacian in Fourier space
    kLaplace = kMatx*kMatx + kMaty*kMaty

    return kMatx, kMaty, kLaplace


###########################################################################
#
# Function to solve poisson problem, Laplacian(psi) = -Vorticity
#
###########################################################################

def please_Solve_Poission(w_hat,kx,ky,NX,NY):

 

    psi_hat = np.zeros((NX,NY),dtype=np.complex_)  # Initialize solution matrix
    kVecX = kx[0,:]                                # Gives row vector from kx
    kVecY = ky[:,0]                                # Gives column vector from ky

    for i in range(1,NX+1):
        for j in range(1,NY+1):
            if ( i+j > 2 ):
                psi_hat[i-1,j-1] = -w_hat[i-1,j-1] / ( ( kVecX[j-1]**2 + kVecY[i-1]**2 ) ) # "inversion step"

    return psi_hat



###########################################################################
#
# Function to perform one time-step of Crank-Nicholson Semi-Implicit
# timestepping routine to get next time-step's vorticity coefficients in
# fourier (frequency space). 
#
# Note: 1. The nonlinear advection is handled explicitly
#       2. The viscous term is handled implictly
#
###########################################################################

def please_Perform_Crank_Nicholson_Semi_Implict(dt,nu,NX,NY,kLaplace,advect_hat,vort_hat):

    for i in range(1,NX+1):
        for j in range(1,NY+1):

            #Crank-Nicholson Semi-Implicit Time-step
            vort_hat[i-1,j-1] = ( (1 + dt/2*nu*kLaplace[i-1,j-1] )*vort_hat[i-1,j-1] - dt*advect_hat[i-1,j-1] ) / (  1 - dt/2*nu*kLaplace[i-1,j-1] )

    return vort_hat




###########################################################################
#
# Function that performs the SIMULATION!
#
###########################################################################


def FFT_NS_Solver():

#
# Solves the Navier-Stokes equations in the Vorticity-Stream Function
# formulation using a pseudo-spectral approach w/ FFT
#
# Author: Nicholas A. Battista
# Created: Novermber 29, 2014
# Modified: September 11, 2019
# 
# Equations of Motion:
# D (Vorticity) /Dt = nu*Laplacian(Vorticity)  
# Laplacian(Psi) = - Vorticity                                                       
#
#      Real Space Variables                   Fourier (Frequency) Space                                                          
#       SteamFunction: Psi                     StreamFunction: Psi_hat
# Velocity: u = Psi_y & v = -Psi_x              Velocity: u_hat ,v_hat
#         Vorticity: Vort                        Vorticity: Vort_hat
#
#
# IDEA: for each time-step
#       1. Solve Poisson problem for Psi (in Fourier space)
#       2. Compute nonlinear advection term by finding u and v in real
#          variables by doing an inverse-FFT, compute advection, transform
#          back to Fourier space
#       3. Time-step vort_hat in frequency space using a semi-implicit
#          Crank-Nicholson scheme (explicit for nonlinear adv. term, implicit
#          for viscous term)
#

    # Print key fluid solver ideas to screen
    print_FFT_NS_Info()

    #
    # Simulation Parameters
    #
    nu=1.0e-3  # kinematic viscosity
    NX = 256   # # of grid points in x
    NY = 256   # # of grid points in y
    LX = 1.0   # 'Length' of x-Domain
    LY = 1.0   # 'Length' of y-Domain

    #
    # Choose initial vorticity state
    # Choices:  'half', 'qtrs', 'rand', 'bubble1', 'bubbleSplit', 'bubble3', 'jets'
    #
    choice='bubble3'
    vort_hat,dt,tFinal,plot_dump = please_Give_Initial_Vorticity_State(choice,NX,NY)

    #
    # Initialize wavenumber storage for fourier exponentials
    #
    kMatx, kMaty, kLaplace = please_Give_Wavenumber_Matrices(NX,NY)


    t=0.0                                                 # Initialize time to 0.0
    nTot = int(tFinal/dt)                                 # Total number of time-steps
    print('Simulation Time: {0:6.6f}\n'.format(t))        # Print initial time
    for n in range(0,nTot+1):                             # Enter Time-Stepping Loop!

        # Printing zero-th time-step
        if n==0:

            #Solve Poisson Equation for Stream Function, psi
            psi_hat = please_Solve_Poission(vort_hat,kMatx,kMaty,NX,NY)

            #Find Velocity components via derivatives on the stream function, psi
            u  = fftpack.ifft2( kMaty*psi_hat ).real        # Compute  y derivative of stream function ==> u = psi_y
            v  = fftpack.ifft2(-kMatx*psi_hat ).real        # Compute -x derivative of stream function ==> v = -psi_x

            # SAVING DATA TO VTK #
            ctsave = 0  # total # of time-steps
            pSave = 0   # time-step data counter

            # CREATE VTK DATA FOLDER 
            try:
                os.mkdir('vtk_data')
            except FileExistsError:
                #File already exists
                pass

            # Transform back to real space via Inverse-FFT
            vort_real = fftpack.ifft2(vort_hat).real

            # Save .vtk data!
            # Note: switch order of u and v in this function bc of notation-> f(x,y) here rather than matrix convention of y(row,col) w/ y=row, x=col
            print_vtk_files(ctsave,v,u,vort_real,LX,LY,NX,NY)

        else:

            #Solve Poisson Equation for Stream Function, psi
            psi_hat = please_Solve_Poission(vort_hat,kMatx,kMaty,NX,NY)

            #Find Velocity components via derivatives on the stream function, psi
            u  = fftpack.ifft2( kMaty*psi_hat ).real        # Compute  y derivative of stream function ==> u = psi_y
            v  = fftpack.ifft2(-kMatx*psi_hat ).real        # Compute -x derivative of stream function ==> v = -psi_x

            #Compute derivatives of voriticty to be "advection operated" on
            vort_X = fftpack.ifft2( kMatx*vort_hat ).real  # Compute  x derivative of vorticity
            vort_Y = fftpack.ifft2( kMaty*vort_hat ).real  # Compute  y derivative of vorticity

            #Compute nonlinear part of advection term
            advect = u*vort_X + v*vort_Y           # Advection Operator on Vorticity: (u,v).grad(vorticity)   
            advect_hat = fftpack.fft2(advect)      # Transform advection (nonlinear) term of material derivative to frequency space

            # Compute Solution at the next step (uses Crank-Nicholson Time-Stepping)
            vort_hat = please_Perform_Crank_Nicholson_Semi_Implict(dt,nu,NX,NY,kLaplace,advect_hat,vort_hat)
            #vort_hat = ((1/dt + 0.5*nu*kLaplace)./(1/dt - 0.5*nu*kLaplace)).*vort_hat - (1./(1/dt - 0.5*nu*kLaplace)).*advect_hat

            # Update time
            t=t+dt 

            # Save files info!
            ctsave = ctsave + 1
            if ( ctsave % plot_dump == 0 ):

                # increment Print counter
                pSave = pSave + 1

                # Transform back to real space via Inverse-FFT
                vort_real = fftpack.ifft2(vort_hat).real

                # Save .vtk data!
                # Note: switch order of u and v in this function bc of notation-> f(x,y) here rather than matrix convention of y(row,col) w/ y=row, x=col
                print_vtk_files(pSave,v,u,vort_real,LX,LY,NX,NY)

                # Plot simulation time
                print('Simulation Time(s): {0:6.6f}\n'.format(t))


    # ENDS TIME-LOOP!

if __name__ == "__main__":
    FFT_NS_Solver()
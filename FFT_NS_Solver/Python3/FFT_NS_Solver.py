'''Solves the Navier-Stokes equations in the Vorticity-Stream Function
 formulation using a pseudo-spectral approach w/ FFT

 Author: Nicholas A. Battista
 Created: Novermber 29, 2014 (MATLAB VERSION)
 Created: April 27, 2017 (PYTHON3 VERSION)
 
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
from print_vtk_files import *

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
    print(' Modified: September 11, 2019 \n\n')
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


    elif ( choice == 'bubble2' ):

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

        buff = 4 # the # of grid cells around each vortex region (make sure even)
        vort=np.zeros([NX,NY])
        vort[0+buff:int(NX/2)-int(buff/2),0+buff:-buff]=1
        vort[int(NX/2)+0+int(buff/2):-buff,0+buff:-buff]=1
        dt=1e-2        # time step
        tFinal = 5     # final time
        plot_dump=20   # interval for plots

    elif ( choice == 'qtrs' ):

        #
        # USE SQUARE: Lx = Ly, Nx = Ny
        #

        vort = -0.25*np.ones([NX,NY])
        vort[1:int(NX/2)-1,1:int(NY/2)-1]=0.25
        vort[int(NX/2):,int(NY/2):]=0.25
        dt=1e-2      # time step
        tFinal=5   # final time
        plot_dump=20  # interval for plots'

    elif ( choice == 'rand' ):

        #
        # Any domain is fine, as long as Lx/Nx = Ly/Ny
        #

        vort = 2*np.random.rand(NX,NY)-1
        dt=1e-1       # time step
        tFinal = 1000 # final time
        plot_dump=25  # interval for plots
    
    # WILL WORK ON OTHER EXAMPLES ONCE CODE IS UP AND RUNNING!
    '''
    elseif strcmp(choice,'bubble1')

        #
        # USE SQUARE: Lx = Ly, Nx = Ny
        #

        #radius of bubble (centered in middle of domain, given in terms of mesh widths)
        radius = 0.25*(NX/2)^2;

        vort = 0.25*rand(NX,NY)-0.50
        a=repmat(-NX/4+1:NX/4,[NY/2 1])
        b1 = ( (a-1).^2 +  (a+1)'.^2 ) < radius
        b1 = double(b1)
        [r1,c1]=find(b1==1)
        b1 = 0.5*rand(NX/2,NY/2)-0.25
        for i=1:length(r1)
            b1(r1(i),c1(i))=  0.5*(rand(1)+1)
        end
        vort(NX/4+1:3*NX/4,NY/4+1:3*NY/4) = b1

        dt=5e-3      # time step
        tFinal = 7.5   # final time
        plot_dump=10 # interval for plots

    elseif strcmp(choice,'bubbleSplit')

        #
        # USE SQUARE: Lx = Ly, Nx = Ny
        #

        # radius of bubble (centered in middle of domain, given in terms of mesh widths)
        radius = 0.25*(NX/2)^2;

        vort = 0.5*rand(NX,NY)-0.25
        a=repmat(-NX/4+1:NX/4,[NY/2 1])
        b1 = ( (a-1).^2 +  (a+1)'.^2 ) < radius
        b1 = double(b1)
        [r1,c1]=find(b1==1)
        b1 = 0.5*rand(NX/2,NY/2)-0.25
        for i=1:length(r1)
            if c1(i) < NX/4
                b1(r1(i),c1(i))=  0.10*(rand(1)-1.0)
            else
                b1(r1(i),c1(i))=  0.10*(rand(1)+0.90)
            end
        end
        vort(NX/4+1:3*NX/4,NY/4+1:3*NY/4) = b1

        dt=5e-3      # time step
        tFinal = 7.5 # final time
        plot_dump=10 # interval for plots

    elseif strcmp(choice,'bubble2')

        #
        # USE SQUARE: Lx = Ly, Nx = Ny
        #

        # radius of bubble (centered in middle of domain, given in terms of mesh widths)
        radius1 = 0.25*(NX/2)^2;
        radius2 = 0.175*(NX/2)^2;
        radius3 = 0.10*(NX/2)^2;

        #Initialize vort matrix
        vort = 2*rand(NX,NY)-1

        ex = 0; #Makes sure full bubbles
        sL = 0; #shift left
        a1=repmat(-NX/4+(1-ex):NX/4,[NY/2+ex 1])
        b1 = ( (a1-1).^2 +  (a1+1)'.^2 ) < radius1 #NX*8+NX/1.5
        b1 = double(b1)
        nZ = find(b1)
        b1(nZ) = 0.8 
        [r1,c1]=find(b1==0)
        for i=1:length(r1)
            b1(r1(i),c1(i))=  2*rand(1)-1
        end
        vort(NX/4+(1-ex):3*NX/4,NY/4+(1-ex)-sL:3*NY/4-sL) = b1

        ex = 0;  #Makes sure full bubbles
        sL = 20; #shift left
        a2=repmat(-NX/8+(1-ex):NX/8,[NY/4+ex 1])
        b2 = ( (a2-1).^2 +  (a2+1)'.^2 ) < radius2 #2*NX+NX/0.75
        b2 = double(b2)
        nZ = find(b2)
        b2(nZ) = -1.0 
        [r2,c2]=find(b2==0)
        for i=1:length(r2)
            b2(r2(i),c2(i))=  1.0 
        end
        vort(3*NX/8+(1-ex):5*NX/8,3*NY/8+(1-ex)-sL:5*NY/8-sL) = b2


        ex = 2; #Makes sure full bubbles
        sR = 8; #shift right
        sD = 25;#shift down        
        a3=repmat(-NX/16+(1-ex):NX/16,[NY/8+ex 1])
        b3 = ( (a3-1).^2 +  (a3+1)'.^2 ) < radius3 #NX/2
        b3 = double(b3)
        nZ=find(b3)
        b3(nZ)= 1.0 
        [r3,c3]=find(b3==0)
        for j=1:length(r3)
            b3(r3(j),c3(j)) =  -1.0 
        end
        vort(7*NX/16+(1-ex)-sD:9*NX/16-sD,7*NY/16+(1-ex)+sR:9*NY/16+sR) = b3

        dt = 1e-2      # time step
        tFinal = 30    # final time
        plot_dump= 50  # interval for plots

    end''' 

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
    NX = 512   # # of grid points in x
    NY = 256   # # of grid points in y
    LX = 1     # 'Length' of x-Domain
    LY = 0.5   # 'Length' of y-Domain

    #
    # Choose initial vorticity state
    # Choices:  'half', 'qtrs', 'rand' (to be implemented: 'bubble1', 'bubble2', 'bubbleSplit', see MATLAB version)
    #
    choice='half'
    vort_hat,dt,tFinal,plot_dump = please_Give_Initial_Vorticity_State(choice,NX,NY)

    #
    # Initialize wavenumber storage for fourier exponentials
    #
    kMatx, kMaty, kLaplace = please_Give_Wavenumber_Matrices(NX,NY)


    t=0.0                                                 # Initialize time to 0.0
    nTot = int(tFinal/dt)                                 # Total number of time-steps
    print('Simulation Time(s): {0:6.6f}\n'.format(t))     # Print initial time
    for n in range(0,nTot+1):                             # Enter Time-Stepping Loop!

        # Printing zero-th time-step
        if n==0:

            #Solve Poisson Equation for Stream Function, psi
            psi_hat = please_Solve_Poission(vort_hat,kMatx,kMaty,NX,NY)

            #Find Velocity components via derivatives on the stream function, psi
            u  = fftpack.ifft2( kMaty*psi_hat ).real        # Compute  y derivative of stream function ==> u = psi_y
            v  = fftpack.ifft2(-kMatx*psi_hat ).real        # Compute -x derivative of stream function ==> v = -psi_x

            # SAVING DATA TO VTK #
            ctsave = 0

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

                # Transform back to real space via Inverse-FFT
                vort_real = fftpack.ifft2(vort_hat).real

                # Save .vtk data!
                # Note: switch order of u and v in this function bc of notation-> f(x,y) here rather than matrix convention of y(row,col) w/ y=row, x=col
                print_vtk_files(ctsave,v,u,vort_real,LX,LY,NX,NY)

                # Plot simulation time
                print('Simulation Time(s): {0:6.6f}\n'.format(t))


    # ENDS TIME-LOOP!

if __name__ == "__main__":
    FFT_NS_Solver()
###########################################################################
#
# FUNCTION: from a u.XXXX.vtk file, we will extract U,V (x-Velocity and
# y-Velocity components), compute the background vorticity, and then return
# it to initialize a simulation.
#
###########################################################################

import numpy as np
import os

#################################################################################
#
# FUNCTION: Reads in (x,y) positions of the immersed boundary from .vtk format
#          
################################################################################

def read_Eulerian_Velocity_Field_vtk(simNums):
    
    simulation_path = os.getcwd();  # Stores working directory's path        
    velocity_path = 'Get_Initial_Vorticity_From_Stored_Vector_Field'
    os.chdir(velocity_path);        # cd's into vorticity initialization folder

    filename = 'u.' + str(simNums) + '.vtk'

    # Stores grid resolution from .vtk file
    Nx = np.genfromtxt(filename, skip_header=5, usecols=(1),max_rows=1)   
    
    # Stores desired Eulerian data
    e_data_X = np.genfromtxt(filename, skip_header=13, usecols=range(0,3*int(Nx),3),max_rows=int(Nx))
    e_data_Y = np.genfromtxt(filename, skip_header=13, usecols=range(1,3*int(Nx),3),max_rows=int(Nx))

    # Go back to working directory
    os.chdir(simulation_path);     

    return e_data_X, e_data_Y

############################################################################
#
# FUNCTION: computes the VORTICITY from the vector field's components, U and V
#           e.g., vec{u} = (U,V)
#
##############################################################################

def compute_Vorticity(U,V,dx,dy):

    # Get Size of Matrix
    [Ny,Nx] = U.shape

    # Initialize storage matrices
    Uy = np.zeros((Ny,Nx))
    Vx = np.zeros((Ny,Nx))

    #
    # Find partial derivative with respect to x of vertical component of
    # velocity, V
    #
    # Recall '-1' equivalent to MATLAB's 'end'
    #
    for i in range(1,Nx+1):
        
        ii=i-1 # index starting at 0

        if (i==1):
            Vx[0:,ii] = ( V[0:,1] - V[0:,-1] ) / (2*dx)
        elif (i==Nx):
            Vx[0:,ii] = ( V[0:,0] - V[0:,ii-1] ) / (2*dx)
        else:
            Vx[0:,ii] = ( V[0:,ii+1] - V[0:,ii-1] ) / (2*dx)
        

    #
    # Find partial derivative with respect to y of horizontal component of
    # velocity, U
    #
    for j in range(1,Ny+1):

        jj=j-1 # index starting at 0

        if (j==1):
            Uy[jj,0:] = ( U[1,0:] - U[-1,0:] ) / (2*dy)
        elif (j==Ny):
            Uy[jj,0:] = ( U[0,0:] - U[jj-1,0:] ) / (2*dy)
        else:
            Uy[jj,0:] = ( U[jj+1,0:] - U[jj-1,0:] ) / (2*dy)



    # Take difference to find vorticity ( Curl(vec{u}) in 2D )
    vorticity = Vx - Uy

    return vorticity


###########################################################################
#
# FUNCTION: from a u.XXXX.vtk file, we will extract U,V (x-Velocity and
# y-Velocity components), compute the background vorticity, and then return
# it to initialize a simulation.
#
###########################################################################

def give_Initial_Vorticity_From_Velocity_Field_VTK_File():

    #
    # Grid Parameters (Make sure to match from FFT_NS_Solver.m)
    #
    Lx = 1     # x-Length of Computational Grid
    Ly = 1     # y-Length of Computational Grid
    Nx = 256   # Grid resolution in x-Direction
    Ny = 256   # Grid resolution in y-Direction
    dx = Lx/Nx # Spatial step-size in x
    dy = Ly/Ny # Spatial step-size in y


    #
    # Get back components of velocity field, U,V, e.g., \vec{u} = (U,V)
    # 
    # Note: U,V are both matrices
    #
    simNumsString = '0070'
    U_orig,V_orig = read_Eulerian_Velocity_Field_vtk(simNumsString)

    # Note the u.0070.vtk case was performed at 512x512 resolution, hence we
    # will downsample for the velocity field, e.g., for 256x256 choose every
    # other point
    lenY,lenX = U_orig.shape
    
    # Amount to downsample by
    downsample = int(lenY/Ny)

    # Store newly downsampled matrices
    U = U_orig[::downsample,::downsample]
    V = V_orig[::downsample,::downsample]

    #
    # Compute Vorticity from Velocity Components
    #
    initial_vorticity = compute_Vorticity(U,V,dx,dy)

    #
    # Transpose the vorticity
    #
    initial_vorticity = np.transpose(initial_vorticity)

    return initial_vorticity

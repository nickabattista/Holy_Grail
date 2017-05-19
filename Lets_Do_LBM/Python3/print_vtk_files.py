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

import os
import numpy as np

##############################################################################
#
# FUNCTION: gives appropriate string number for filename in printing the
#           .vtk data files.
#
##############################################################################

def give_String_Number_For_VTK(num):

    #num: # of file to be printed

    if (num < 10):
        strNUM = '000' + np.str(num)
    elif (num < 100):
        strNUM = '00' + np.str(num)
    elif (num<1000):
        strNUM = '0' + np.str(num)
    else:
        strNUM = np.str(num)
    
    return strNUM


##############################################################################
#
# FUNCTION: prints 3D scalar matrix to VTK formated file
#
# Author: Nicholas A. Battista
# Date: 8/24/16
# Github: http://github.org/nickabattista
# Institution: UNC-CH
# Lab: Laura Miller Lab
#
##############################################################################

def savevtk_scalar(array, filename, colorMap,dx,dy):
    ''' Prints scalar matrix to vtk formatted file.
    
    Args:
        array: 2-D ndarray
        filename: file name
        colorMap:
        dx:
        dy:'''
    
    # array is matrix of size 'nx x ny x nz' containing scalar data on
    #              computational grid
    # filename:    What you are saving the VTK file as (string)
    # colorMap:    What you are naming the data you're printing (string)
    # dx,dy:       Grid spacing (resolution)

    #  	Note:
    #   3D is clearly broken in this code, but there were still some reminants 
    #   in the matlab version. Given the choice of doing try/except blocks to
    #   keep these reminants or to kill them entirely, I'm choosing to kill them.
    #   So, specifically, nz is now gone. I will keep the output the same,
    #   however, for compatibility. So 1 will be printed in the Z column.

    nx,ny = array.shape

    with open(filename,'w') as fid:
        fid.write('# vtk DataFile Version 2.0\n')
        fid.write('Comment goes here\n')
        fid.write('ASCII\n')
        fid.write('\n')
        fid.write('DATASET STRUCTURED_POINTS\n')
        # 1 below was nz
        fid.write('DIMENSIONS    {0}   {1}   {2}\n'.format(nx, ny, 1))
        fid.write('\n')
        fid.write('ORIGIN    0.000   0.000   0.000\n')
        fid.write('SPACING   '+str(dx)+str(' ')+str(dy)+'   1.000\n')
        fid.write('\n')
        # The 1 below was nz
        fid.write('POINT_DATA   {0}\n'.format(nx*ny*1))
        fid.write('SCALARS '+colorMap+' double\n')
        fid.write('LOOKUP_TABLE default\n')
        fid.write('\n')
        for b in range(ny):
            for c in range(nx):
                fid.write('{0} '.format(array[c,b]))
            fid.write('\n')
    #Python 3.5 automatically opens in text mode unless otherwise specified    


##############################################################################
#
# FUNCTION: prints structured point vector data to vtk formated file
#
# Author: Nicholas A. Battista
# Date: 8/24/16
# Github: http://github.org/nickabattista
# Institution: UNC-CH
# Lab: Laura Miller Lab
#
##############################################################################

def savevtk_vector(X, Y, filename, vectorName,dx,dy):
    ''' Prints matrix vector data to vtk formated file.
    
    Args:
        X: 2-D ndarray
        Y: 2-D ndarray
        filename: file name
        vectorName:
        dx:
        dy:'''
    
    # X,Y is matrix of size Nx3 containing X,Y-direction vector components
    #              Col 1: x-data
    #              Col 2: y-data
    #              Col 3: z-data
    # filename:    What you are saving the VTK file as (string)
    # vectorName:  What you are naming the data you're printing (string)
    # dx,dy:       Grid spacing (resolution)

    #   Note:
    #   3D is clearly broken in this code, but there were still some reminants 
    #   in the matlab version. Given the choice of doing try/except blocks to
    #   keep these reminants or to kill them entirely, I'm choosing to kill them.
    #   So, specifically, nz is now gone. I will keep the output the same,
    #   however, for compatibility. So 1 will be printed in the Z column.
    
    #   Checks for compatibility of array sizes
    assert (X.shape == Y.shape), 'Error: velocity arrays of unequal size'
    nx, ny = X.shape
    
    XRow = X.shape[0]
    XCol = X.shape[1]
    YRow = Y.shape[0]
    YCol = Y.shape[1]

    with open(filename,'w') as fid:
        fid.write('# vtk DataFile Version 2.0\n')
        fid.write('Comment goes here\n')
        fid.write('ASCII\n')
        fid.write('\n')
        fid.write('DATASET STRUCTURED_POINTS\n')
        # 1 below was nz
        fid.write('DIMENSIONS    {0}   {1}   {2}\n'.format(nx, ny, 1))
        fid.write('\n')
        fid.write('ORIGIN    0.000   0.000   0.000\n')
        #fid.write('SPACING   1.000   1.000   1.000\n') #if want [1,32]x[1,32] rather than [0,Lx]x[0,Ly]
        fid.write('SPACING   '+str(dx)+str(' ')+str(dy)+'   1.000\n')
        fid.write('\n')
        fid.write('POINT_DATA   {0}\n'.format(nx*ny))
        fid.write('VECTORS '+vectorName+' double\n')
        fid.write('\n')
        for b in range(ny):
            for c in range(nx):
                fid.write('{0} '.format(X[c,b]))
                fid.write('{0} '.format(Y[c,b]))
                fid.write('0 ')
            fid.write('\n')
    #Python 3.5 automatically opens in text mode unless otherwise specified



##############################################################################
#
# FUNCTION: prints matrix vector data to vtk formated file
#
##############################################################################

def savevtk_points( X, filename, vectorName):
    ''' Prints matrix vector data to vtk formated file
    
    Args:
        X: Matrix of size Nx3
        filename:
        vectorName:'''

    N = X.shape[0]

    with open(filename,'w') as file:
        file.write('# vtk DataFile Version 2.0\n')
        file.write(vectorName+'\n')
        file.write('ASCII\n')
        file.write('DATASET UNSTRUCTURED_GRID\n\n')
        file.write('POINTS {0} float\n'.format(N))
        for ii in range(N):
            file.write('{0:.15e} {1:.15e} {2:.15e}\n'.format(X[ii,0],X[ii,1],0))
        file.write('\n')
        #
        #First: # of "Cells", Second: Total # of info inputed following
        file.write('CELLS {0} {1}\n'.format(N,2*N))
        for s in range(N):
            file.write('{0} {1}\n'.format(1,s))
        file.write('\n')
        #
        file.write('CELL_TYPES {0}\n'.format(N)) # N = # of "Cells"
        for ii in range(N):
           file.write('1 ')
        file.write('\n')









##############################################################################
#
# FUNCTION: gives appropriate string number for filename in printing the
# .vtk files.
#
##############################################################################

def print_vtk_files(ctsave,U,V,vorticity,Lx,Ly,nx,ny,Bound_Pts):

    #Give spacing for grid
    dx = Lx/(nx-1) 
    dy = Ly/(ny-1)

    #Go into vtk_data directory. This was throwing an error because we're already there!
    if os.path.split(os.getcwd())[1] != 'vtk_data':
        os.chdir('vtk_data')

    #Find string number for storing files
    strNUM = give_String_Number_For_VTK(ctsave)

    #Print Boundary Pts to .vtk format
    lagPtsName = 'Bounds.'+strNUM+'.vtk'
    savevtk_points(Bound_Pts, lagPtsName, 'Bounds')

    #Prints x-Velocity Component
    confName = 'uX.' + strNUM + '.vtk'
    savevtk_scalar(U, confName, 'uX',dx,dy)

    #Prints y-Velocity Component
    confName = 'uY.' + strNUM + '.vtk'
    savevtk_scalar(V, confName, 'uY',dx,dy)

    #Prints Mag. of Velocity 
    confName = 'uMag.' + strNUM + '.vtk'
    uMag = np.sqrt(U*U + V*V)
    savevtk_scalar(uMag, confName, 'uMag',dx,dy)

    #Prints Vorticity
    confName = 'Omega.' + strNUM + '.vtk'
    savevtk_scalar(vorticity, confName, 'Omega',dx,dy)

    #Print VECTOR DATA (i.e., velocity data) to .vtk file
    velocityName = 'u.' + strNUM + '.vtk' 
    savevtk_vector(U, V, velocityName, 'u',dx,dy)

    #Get out of viz_IB2d folder
    os.chdir('..')
'''
     This script solves the FitzHugh-Nagumo Equations in 1d, which are 
     a simplified version of the more complicated Hodgkin-Huxley Equations. 
    
     Author:  Nick Battista
     Email: nickabattista@gmail.com
     Institution: UNC-CH
     Created: 09/11/2015 (MATLAB)
     Created: 04/26/2017 (Python3)
    
     Equations:
     dv/dt = D*Laplacian(v) + v*(v-a)*(v-1) - w - I(t)
     dw/dt = eps*(v-gamma*w)
    
     Variables & Parameters:
     v(x,t): membrane potential
     w(x,t): blocking mechanism
     D:      diffusion rate of potential
     a:      threshold potential
     gamma:  resetting rate
     eps:    strength of blocking
     I(t):   initial condition for applied activation
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings


###########################################################################
#
# FUNCTION: gives Laplacian of the membrane potential, note: assumes
# periodicity and uses the 2nd order central differencing operator.
#
###########################################################################

def give_Me_Laplacian(v,dx):

    Npts = v.size
    DD_v = np.zeros(Npts)

    for ii in range(1,Npts+1):
       i=ii-1
       if (ii==1):
           DD_v[i] = ( v[i+1] - 2*v[i] + v[-1] ) / dx**2
       elif (ii == Npts):
           DD_v[i] = ( v[1] - 2*v[i] + v[i-1] ) / dx**2
       else:
           DD_v[i] = ( v[i+1] - 2*v[i] +  v[i-1] ) /dx**2
    
    return DD_v

###########################################################################
#
# FUNCTION: the injection function, Iapp = activation wave for system, and
# returns both the activation as well as updated pulse_time
#
###########################################################################

def Iapp(pulse_time,i1,i2,I_mag,N,pulse,dp,t,app):


    #Check to see if there should be a pulse
    if t > (pulse_time):

        # Sets pulsing region to current amplitude of I_mag x\in[i1*N,i2*N]
        for jj in range(int(np.floor(i1*N)),int(np.floor(i2*N)+1)):
            j=jj-1
            app[j] = I_mag  
        
        # Checks if the pulse is over & then resets pulse_time to the next pulse time.
        if ( t > (pulse_time+dp) ):
            pulse_time = pulse_time+pulse
        
    else:

        # Resets to no activation
        app = np.zeros(N+1)
    
    
    return app,pulse_time    



###########################################################################
#
# Function that actually performs the simulation time-stepping routine
#
###########################################################################

def FitzHugh_Nagumo_1d():

    # This script solves the FitzHugh-Nagumo Equations in 1d, which are 
    # a simplified version of the more complicated Hodgkin-Huxley Equations. 
    #
    # Author:  Nick Battista
    # Email: nickabattista@gmail.com
    # Institution: UNC-CH
    # Created: 09/11/2015 (MATLAB)
    # Created: 04/26/2017 (Python 3)
    #
    # Equations:
    # dv/dt = D*Laplacian(v) + v*(v-a)*(v-1) - w - I(t)
    # dw/dt = eps*(v-gamma*w)
    #
    # Variables & Parameters:
    # v(x,t): membrane potential
    # w(x,t): blocking mechanism
    # D:      diffusion rate of potential
    # a:      threshold potential
    # gamma:  resetting rate
    # eps:    strength of blocking
    # I(t):   initial condition for applied activation

    # Parameters in model #
    D = 1.0        # Diffusion coefficient
    a = 0.3        # Threshold potential (Note: a=0.3 is traveling wave value, a=0.335 is interesting)
    gamma = 1.0    # Resetting rate (Note: large values give 'funky thick' traveling wave, gamma = 1.0 is desired)
    eps = 0.001    # Blocking strength (Note: eps = 0.001 is desired)
    I_mag = 0.05   # Activation strength

    # Discretization/Simulation Parameters #
    N = 800                    # # of discretized points  
    L = 2000                   # Length of domain, [0,L]
    dx = L/N                   # Spatial Step
    x = np.arange(0,L+dx,dx)   # Computational Domain

    # Temporal  Parameters #
    T_final = 10000         # Sets the final time
    Np = 10                 # Set the number of pulses
    pulse = T_final/Np      # determines the length of time between pulses.
    NT = 800000             # Number of total time-steps to be taken
    dt = T_final/NT         # Time-step taken
    i1 = 0.475              # fraction of total length where current starts
    i2 = 0.525              # fraction of total length where current ends
    dp = pulse/50           # Set the duration of the current pulse
    pulse_time = 0          # pulse time is used to store the time that the next pulse of current will happen
    IIapp=np.zeros(N+1)     # this vector holds the values of the applied current along the length of the neuron
    dptime = T_final/100    # This sets the length of time frames that are saved to make a movie.

    # Initialization #
    v = np.zeros(N+1)
    w = v
    t=0
    ptime = 0       
    tVec = np.arange(0,T_final,dt)
    Nsteps = tVec.size
    vNext = np.zeros((Nsteps,N+1)) 
    vNext[0,:] = v.transpose()
    wNext = np.zeros((Nsteps,N+1)) 
    wNext[0,:] = w.transpose()

    #
    # **** # **** BEGIN SIMULATION! **** # **** #
    #
    for i in range(1,Nsteps):

         # Update the time
        t = t+dt                        

        # Give Laplacian
        DD_v_p = give_Me_Laplacian(v,dx)  

        # Gives activation wave
        IIapp,pulse_time = Iapp(pulse_time,i1,i2,I_mag,N,pulse,dp,t,IIapp)

        # Update potential and blocking mechanism, using Forward Euler
        vN = v + dt * ( D*DD_v_p - v*(v-a)*(v-1) - w + IIapp )
        wN = w + dt * ( eps*( v - gamma*w ) )

        # Update time-steps
        v = vN
        w = wN

        # Store time-step values
        vNext[i,:] = v.transpose()
        wNext[i,:] = w.transpose()

        #This is used to determine if the current time step will be a frame in the movie
        if ( t > ptime ):

            plt.clf() #Clear previous plots :)

            # Plot Current Time in Simulation
            print('Time(s): {0:6.6f}\n'.format(t))

            # Plot Simulation Data
            plt.figure(1)
            plt.plot(x, v,'r-',linewidth=5)
            plt.axis([0,L,-0.2,1.0])
            plt.xlabel('Distance (x)')
            plt.ylabel('Electropotenital (v)')
            ptime = ptime+dptime
            time.sleep(0.1)                         # Pause for 0.1s
   
            plt.box(on=True)
            plt.draw()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.pause(0.0001) #no idea why this is necessary, but it is


    # ENDS TIME-STEPPING ROUTINE


##############################################################################



if __name__ == "__main__":
    FitzHugh_Nagumo_1d()
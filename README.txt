%*****************************************************************************%
%********************************% HOLY GRAIL %*******************************%
%*****************************************************************************%

 HOLY GRAIL is a collection of various fluid solvers, with built in examples.
   The codes are used mostly for educational and recreational purposes.

 Author: Nicholas A. Battista
 Email:  nick.battista@unc.edu
 Date Created: 2014
 Institution: University of North Carolina at Chapel Hill
 Website: http://battista.web.unc.edu
 GitHub: http://www.github.com/nickabattista

%*****************************************************************************%
%******************************% FLUID SOLVERS %******************************%
%*****************************************************************************%

PROJECTION:

      Solves the Navier-Stokes equations in the Velocity-Pressure formulation 
        using a predictor-corrector projection method approach
 
      Equations of Motion:
            Du/Dt = -Nabla(u) + nu*Laplacian(u)  [Conservation of Momentum]
            Nabla \cdot u = 0                    [Conservation of Mass]                                   

      IDEA: for each time-step
            1. Compute an intermediate velocity field explicitly, 
               use the momentum equation, but ignore the pressure gradient term
            2. Solve the Poisson problem for the pressure, whilst enforcing
               that the true velocity is divergence free. 
            3. Projection Step: correct the intermediate velocity field to
               obtain a velocity field that satisfies momentum and
               incompressiblity.

PSEUDO-SPECTRAL (FFT):
   
      Solves the Navier-Stokes equations in the Vorticity-Stream Function
        formulation using a pseudo-spectral approach w/ FFT
 
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
                for viscous term)

LATTICE-BOLTZMANN:

                                   D2Q9 Model:

                                   c6  c2   c5
                                     \  |  /  
                                   c3- c9 - c1
                                     /  |  \  
                                   c7  c4   c8

     f_i: the probability for site vec(x) to have a particle heading in
     direction i, at time t. These f_i's are called discretized probability 
     distribution functions

     LBM IDEA: 
               1. At each timestep the particle densities propogate in each 
                  direction (1-8).
               2. An equivalent "equilibrium" density is found
               3. Densities relax towards that state, in proportion governed by 
                  tau (parameter related to viscosity)

     CHOICE OF SIMULATION:
              -> The code is setup to run a few different geometries:
	                     a. Flow in a channel
	                     b. Flow around a cylinder
	                     c. Flow around a few cylinders
	                     d. Flow through one porous layer
	                     e. Flow through multiple porous layers

PANEL METHOD:

        Solves the incompressible potential flow equations in 2D.

        Assumptions for Incompressible Potential Flow
              1. Inviscid
              2. Incompressible div(V) = 0
              3. Irrotational   curl(V) = 0
              4. Steady         partial(u)/partial(t) = 0

        What it does: 
                
                This method finds the lift and drag coefficients around an
                airfoil shape, chosen by the user. It also computes the
                pressure distribution over the airfoil as well.

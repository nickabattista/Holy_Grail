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

PANEL METHOD:



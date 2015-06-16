 Solves the Navier-Stokes equations in the Velocity-Pressure formulation 
 using a predictor-corrector projection method approach

 Author: Nicholas A. Battista
 Created: Novermber 24, 2014
 Modified: December 8, 2014
 
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

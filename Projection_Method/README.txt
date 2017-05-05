 Solves the Navier-Stokes equations in the Velocity-Pressure formulation 
 using a predictor-corrector projection method approach

 Author: Nicholas A. Battista
 Created: Novermber 24, 2014 (MATLAB)
 Modified: December 8, 2014  (MATLAB)
 Created: April 25, 2017     (Python3)
 
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

 TO RUN: 
       MATLAB: type <Projection_Method> into the command window
       Python3: type <run Projection_Method.py>

 VISUALIZE: 
       MATLAB/Python 3: Open .vtk data (in "vtk_data" folder) in either 
                        VisIt or ParaView

       NOTE: MATLAB can plot in MATLAB, as well.
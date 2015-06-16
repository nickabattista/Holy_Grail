  Panel Method: Here used to solve the incompressible potential flow 
                equations in 2D.

  Author: Nicholas A. Battista
  Created: January 6, 2015
  Modified: February 5, 2015

  Assumptions for Incompressible Potential Flow
                1. Inviscid
                2. Incompressible div(V) = 0
                3. Irrotational   curl(V) = 0
                4. Steady         partial(u)/partial(t) = 0

  What it does: 
                This method finds the lift and drag coefficients around an
                airfoil shape, chosen by the user. It also computes the
                pressure distribution over the airfoil as well.

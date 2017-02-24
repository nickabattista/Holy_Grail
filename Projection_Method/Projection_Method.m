function Projection_Method()

%
% Solves the Navier-Stokes equations in the Velocity-Pressure formulation 
% using a predictor-corrector projection method approach
%
% Author: Nicholas A. Battista
% Created: Novermber 24, 2014
% Modified: December 8, 2014
% 
% Equations of Motion:
% Du/Dt = -Nabla(u) + nu*Laplacian(u)  [Conservation of Momentum]
% Nabla \cdot u = 0                    [Conservation of Mass]                                   
%
%
% IDEA: for each time-step
%       1. Compute an intermediate velocity field explicitly, 
%          use the momentum equation, but ignore the pressure gradient term
%       2. Solve the Poisson problem for the pressure, whilst enforcing
%          that the true velocity is divergence free. 
%       3. Projection Step: correct the intermediate velocity field to
%          obtain a velocity field that satisfies momentum and
%          incompressiblity.
%

% Print key fluid solver ideas to screen
print_Projection_Info();
pause();

%
% GRID PARAMETERS %
%
Lx = 2.0;        %Lenght in x
Ly = 1.0;        %Length in y
nx=32;           %Initialize X-Grid (Spatial Resolution in x)
ny=16;           %Initialize Y-Grid (Spatial Resolution in y)
dx=2/nx;         %Spatial Distance Definition
xGrid = 0:dx:Lx; %xGrid
yGrid = 0:dx:Ly; %yGrid


%
% SIMULATION PARAMETERS %
%
mu = 10;         %Fluid DYNAMIC viscosity (kg / m*s)
rho = 1000;      %Fluid density (kg/m^3) 
nu=mu/rho;       %Fluid KINEMATIC viscosity
numPredCorr = 3; %Number of Predictor-Corrector Steps
maxIter=200;     %Maximum Iterations for SOR Method to solve Elliptic Pressure Equation
beta=1.25;       %Relaxation Parameter 


%
% Initialize all storage quantities %
%
[u, v, p, uTemp, vTemp, uAvg, vAvg, vorticity, c] = initialize_Storage(nx,ny);


%
% CHOOSE SIMULATION (gives chosen simulation parameters) %
% Possible choices: 'cavity_left', 'whirlwind', 'twoSide_same', 'twoSide_opp', 'corner'
%
choice = 'twoSide_opp';
[uTop,uBot,vRight,vLeft,dt,nStep,pStep,bVel,xStart,yStart] = please_Give_Me_BCs(choice);


%PRINT SIMULATION INFO %
print_Simulation_Info(choice,dt,dx,nu,bVel,Ly); 


% SAVING DATA TO VTK %
print_dump = 10;
ctsave = 0;
% CREATE VIZ_IB2D FOLDER and VISIT FILES
mkdir('vtk_data');
print_vtk_files(ctsave,u,v,p,vorticity,Lx,Ly,nx,ny);



%
% BEGIN TIME-STEPIN'!
% 
t=0.0; %Initialize time
for j=1:nStep
    
    %Enforce Boundary Conditions (Solve for "ghost velocities")
    u(1:nx+1,1)=2*uBot-u(1:nx+1,2);
    u(1:nx+1,ny+2)=2*uTop-u(1:nx+1,ny+1);
    v(1,1:ny+1)=2*vLeft-v(2,1:ny+1);
    v(nx+2,1:ny+1)=2*vRight-v(nx+1,1:ny+1);

    %Start Predictor-Corrector Steps
    for k=1:numPredCorr
    
        %Find auxillary (temporary) velocity fields for predictor step
        [uTemp, vTemp] = give_Auxillary_Velocity_Fields(dt,dx,nu,nx,ny,u,v,uTemp,vTemp);
    
        %Solve Elliptic Equation for Pressure via SOR scheme
        p = solve_Elliptic_Pressure_Equation(dt,dx,nx,ny,maxIter,beta,c,uTemp,vTemp,p);

        % Velocity Correction
        u(2:nx,2:ny+1)=uTemp(2:nx,2:ny+1)-(dt/dx)*(p(3:nx+1,2:ny+1)-p(2:nx,2:ny+1));
        v(2:nx+1,2:ny)=vTemp(2:nx+1,2:ny)-(dt/dx)*(p(2:nx+1,3:ny+1)-p(2:nx+1,2:ny));
    end
    
    %Update Simulation Time (not needed in algorithm)
    t=t+dt;
    
    % PLOTTING IN MATLAB
    %if ( mod(j,pStep) == 0 )
    %    fprintf('Simulation Time: %d\n',t);
        
        %Plot Timestep Velocity Fields and Vorticity
    %    plot_Velocity_and_Vorticity(dx,nx,ny,xGrid,yGrid,u,v,uAvg,vAvg,vorticity,xStart,yStart);
    %end
    

    % Save files info!
    ctsave = ctsave + 1;
    if mod(ctsave,print_dump) == 0
        vorticity(1:nx+1,1:ny+1)=(u(1:nx+1,2:ny+2)-u(1:nx+1,1:ny+1)-v(2:nx+2,1:ny+1)+v(1:nx+1,1:ny+1))/(2*dx); 
        print_vtk_files(ctsave,u,v,p,vorticity,Lx,Ly,nx,ny);
        fprintf('Simulation Time: %d\n',t);
    end

end %ENDS TIME-STEPPING


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to initialize storage for velocities, pressure, vorticity,
% coeffs., etc.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [u, v, p, uTemp, vTemp, uAvg, vAvg, vorticity, c] = initialize_Storage(nx,ny)

    %Initialize (u,v) velocities, pressure
    u=zeros(nx+1,ny+2);      %x-velocity (u) initially zero on grid
    v=zeros(nx+2,ny+1);      %y-velocity (v) initially zero on grid
    p=zeros(nx+2,ny+2);      %pressure initially zero on grid
    uTemp=zeros(nx+1,ny+2);  %auxillary x-Velocity (u*) field initially zero on grid
    vTemp=zeros(nx+2,ny+1);  %auxillary y-Velocity (v*) field initially zero on grid

    %Initialize quantities for plotting
    uAvg=zeros(nx+1,ny+1);      %uAvg: averaged x-Velocities on grid (smoothing)
    vAvg=zeros(nx+1,ny+1);      %vAvg: averaged y-Velocities on grid (smoothing)
    vorticity=zeros(nx+1,ny+1); %w: Vorticity

    %Coefficients when solving the Elliptic Pressure Equation w/ SOR (so averaging is consistent)
    c=1/4*ones(nx+2,ny+2);   %Interior node coefficients set to 1/4 (all elements exist)
    c(2,3:ny)=1/3;           %BouTopdary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c(nx+1,3:ny)=1/3;        %BouTopdary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c(3:nx,2)=1/3;           %BouTopdary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c(3:nx,ny+1)=1/3;        %BouTopdary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c(2,2)=1/2;              %Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c(2,ny+1)=1/2;           %Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c(nx+1,2)=1/2;           %Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c(nx+1,ny+1)=1/2;        %Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to find auxillary (temporary) velocity fields in predictor step
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [uTemp, vTemp] = give_Auxillary_Velocity_Fields(dt,dx,nu,nx,ny,u,v,uTemp,vTemp)
    
    %Find Temporary u-Velocity Field
    for i=2:nx 
        for j=2:ny+1 
            uTemp(i,j)=u(i,j)+dt*(-(0.25/dx)*((u(i+1,j)+u(i,j))^2-(u(i,j)+u(i-1,j))^2+(u(i,j+1)+u(i,j))*(v(i+1,j)+v(i,j))-(u(i,j)+u(i,j-1))*(v(i+1,j-1)+v(i,j-1)))+(nu/dx^2)*(u(i+1,j)+u(i-1,j)+u(i,j+1)+u(i,j-1)-4*u(i,j)));
        end
    end

    %Find Temporary v-Velocity Field
    for i=2:nx+1
        for j=2:ny 
            vTemp(i,j)=v(i,j)+dt*(-(0.25/dx)*((u(i,j+1)+u(i,j))*(v(i+1,j)+v(i,j))-(u(i-1,j+1)+u(i-1,j))*(v(i,j)+v(i-1,j))+(v(i,j+1)+v(i,j))^2-(v(i,j)+v(i,j-1))^2)+(nu/dx^2)*(v(i+1,j)+v(i-1,j)+v(i,j+1)+v(i,j-1)-4*v(i,j)));
        end
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to solve elliptic pressure equation using a SOR scheme 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = solve_Elliptic_Pressure_Equation(dt,dx,nx,ny,maxIter,beta,c,uTemp,vTemp,p)

    iter = 1; err = 1; tol = 5e-6;
    pPrev = p;
    while ( (err > tol) && (iter < maxIter) )     
        for i=2:nx+1 
            for j=2:ny+1
                p(i,j)=beta*c(i,j)*(p(i+1,j)+p(i-1,j)+p(i,j+1)+p(i,j-1)-(dx/dt)*(uTemp(i,j)-uTemp(i-1,j)+vTemp(i,j)-vTemp(i,j-1)))+(1-beta)*p(i,j);
            end
        end
        err = max( max( abs( p - pPrev ) ) );
        pPrev = p;
        iter = iter + 1;
    end 
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to plot Velocity, Vorticity, and Streamline Contours at each
% Timestep
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
function plot_Velocity_and_Vorticity(dx,nx,ny,xGrid,yGrid,u,v,uAvg,vAvg,vorticity,xStart,yStart)

    %Plot Info
    xF = xGrid(1);   %Left most x-Point
    xL = xGrid(end); %Right most x-Point
    yF = yGrid(1);   %Bottom most y-Point
    yL = yGrid(end); %Top most y-Point
    
    %Compute x and y directed velocity averages (interpolate from cell-staggered to grid points)
    uAvg(1:nx+1,1:ny+1)=0.5*(u(1:nx+1,2:ny+2)+u(1:nx+1,1:ny+1));
    vAvg(1:nx+1,1:ny+1)=0.5*(v(2:nx+2,1:ny+1)+v(1:nx+1,1:ny+1));

    %Compute magnitude of velocity
    velMag = sqrt( uAvg.*uAvg + vAvg.*vAvg );
    
    %Compute Vorticity
    vorticity(1:nx+1,1:ny+1)=(u(1:nx+1,2:ny+2)-u(1:nx+1,1:ny+1)-v(2:nx+2,1:ny+1)+v(1:nx+1,1:ny+1))/(2*dx); 
    
    %Scale Factor for Velocity Vectors
    scale = 0.19;
    
    % Streamlines Info %
    options = [0.01 10000];
    
    subplot(2,1,1)
    hold off
    contourf(xGrid,yGrid,flipud(rot90(velMag)),7);hold on;
    quiver(xGrid,yGrid,scale*flipud(rot90(uAvg)),scale*flipud(rot90(vAvg)),'r','AutoScale','off');hold on;
    title('Velocity Magnitude Colormap');
    axis equal
    axis([xF xL yF yL]);
    
    subplot(2,1,2)
    hold off
    contourf(xGrid,yGrid,flipud(rot90(vorticity)),3); hold on;
    h = streamline(xGrid,yGrid,flipud(rot90(uAvg)),flipud(rot90(vAvg)),xStart,yStart,options); hold on;
    set(h,'linewidth',1.25,'color','k');
    plot(xStart,yStart,'r.','MarkerSize',6); hold on;
    title('Vorticity Colormap and Streamlines');
    axis equal
    axis([xF xL yF yL]);
    pause(0.01)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function that chooses which simulation to run by changing the BCs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function [uTop,uBot,vRight,vLeft,dt,nStep,printStep,bVel,xStart,yStart] = please_Give_Me_BCs(choice)

% Possible choices: 'cavity_left', 'whirlwind', 'twoSide_same', 'twoSide_opp', 'corner'

if strcmp(choice,'cavity_left')
    
    bVel = 1.0;
    uTop = 0.0; uBot = 0.0; vRight = 0.0; vLeft = bVel;
    
    dt = 0.01;      %Time-step
    nStep=300;      %Number of Time-Steps
    printStep = 5;  %Print ever # of printStep frames
    
    % Streamlines Info %
    xStart = [0.1 0.5 0.7];
    yStart = 0.5*ones(size(xStart));
    
elseif strcmp(choice,'whirlwind')
    
    bVel = 1.0;
    uTop=bVel;  uBot=-bVel; vRight=-bVel; vLeft=bVel;
    
    dt = 0.01;      %Time-step
    nStep=150;      %Number of Time-Steps
    printStep = 2;  %Print ever # of printStep frames
    
    % Streamlines Info %
    yStart = 0.10:0.15:0.40;
    xStart = ones(size(yStart));
    
elseif strcmp(choice,'twoSide_same')
    
    bVel = 2.0;
    uTop = 0.0; uBot = 0.0; vRight = bVel; vLeft = bVel;
    
    dt = 0.01;      %Time-step
    nStep=300;      %Number of Time-Steps
    printStep = 3;  %Print ever # of printStep frames
    
    % Streamlines Info %
    xStart = [0.1 0.5 1.55 1.9];
    yStart = 0.5*ones(size(xStart));

    
elseif strcmp(choice,'twoSide_opp')
    
    bVel = 2.0;
    uTop = 0.0; uBot = 0.0; vRight = -bVel; vLeft = bVel;
    
    dt = 0.01;      %Time-step
    nStep=300;      %Number of Time-Steps
    printStep = 3;  %Print ever # of printStep frames
    
    % Streamlines Info %
    xStart = [0.1 0.5 0.9 1.1 1.55 1.9];
    yStart = 0.5*ones(size(xStart));
    
elseif strcmp(choice,'corner')

    bVel = 1.0;
    uTop = bVel; uBot = 0.0; vRight = 0; vLeft = bVel;
    
    dt = 0.01;      %Time-step
    nStep=300;      %Number of Time-Steps
    printStep = 3;  %Print ever # of printStep frames
    
    % Streamlines Info %
    yStart = 0.10:0.15:0.55;
    xStart = ones(size(yStart));
    
else
   
    fprintf('YOU DID NOT CHOOSE CORRECTLY!!!!!\n');
    fprintf('Simulation DEFAULT: whirlwind\n');
    
    bVel = 1.0;
    uTop=bVel;  uBot=-bVel; vRight=-bVel; vLeft=bVel;
    
    dt = 0.01;      %Time-step
    nStep=150;      %Number of Time-Steps
    printStep = 2;  %Print ever # of printStep frames
    
    % Streamlines Info %
    yStart = 0.10:0.15:0.40;
    xStart = ones(size(yStart));

end

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function that chooses which simulation to run by changing the BCs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_Projection_Info()

fprintf('\n____________________________________________________________________________\n');
fprintf('\nSolves the Navier-Stokes equations in the Velocity-Pressure formulation \n');
fprintf('using a predictor-corrector projection method approach\n\n');
fprintf('Author: Nicholas A. Battista\n');
fprintf('Created: Novermber 24, 2014\n');
fprintf('Modified: December 8, 2014\n');
fprintf('____________________________________________________________________________\n\n');
fprintf('Equations of Motion:\n');
fprintf('Du/Dt = -Nabla(u) + nu*Laplacian(u)  [Conservation of Momentum] \n');
fprintf('Nabla cdot u = 0                     [Conservation of Mass -> Incompressibility]     \n\n');                              
fprintf('IDEA: for each time-step\n');
fprintf('       1. Compute an intermediate velocity field explicitly, \n');
fprintf('          use the momentum equation, but ignore the pressure gradient term\n');
fprintf('       2. Solve the Poisson problem for the pressure, whilst enforcing\n');
fprintf('          that the true velocity is divergence free. \n');
fprintf('       3. Projection Step: correct the intermediate velocity field to\n');
fprintf('          obtain a velocity field that satisfies momentum and\n');
fprintf('          incompressiblity.\n\n');
fprintf('____________________________________________________________________________\n\n');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function that chooses which simulation to run by changing the BCs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_Simulation_Info(choice,dt,dx,nu,bVel,Ly)

% PRINT STABILITY INFO %
fprintf('\nNOTE: dt must be <= %d for STABILITY!\n',0.25*dx^2/nu);
fprintf('Your dt = %d\n\n',dt);

if strcmp(choice,'cavity_left')
    
    fprintf('You are simulating cavity flow\n');
    fprintf('The open cavity is on the left side\n');
    fprintf('Try changing the viscosity or geometry\n\n');
    
elseif strcmp(choice,'whirlwind')
    
    fprintf('You are simulating vortical flow\n');
    fprintf('All velocities on the wall point at the next corner in a CW manner\n');
    fprintf('Try changing the velocity BCs on each wall');
    fprintf('Or try changing the viscosity or geometry\n\n');
    
elseif strcmp(choice,'twoSide_same')
    
    fprintf('You are simulating two-sided cavity flow\n');
    fprintf('The open cavities are on the left and right sides\n');
    fprintf('The vertical velocities n those walls point in the same direction\n');
    fprintf('Try changing the viscosity or geometry\n\n');
    
elseif strcmp(choice,'twoSide_opp')
    
    fprintf('You are simulating two-sided cavity flow\n');
    fprintf('The open cavities are on the left and right sides\n');
    fprintf('The vertical velocities on those walls point in opposite directions\n');
    fprintf('Try changing the viscosity or geometry\n\n');
    
elseif strcmp(choice,'corner')

    fprintf('You are simulating cavity flow, through the left and top boundaries\n');
    fprintf('The vertical velocities on those walls point in opposite directions\n');
    fprintf('Try changing the viscosity or geometry\n\n');
    
else
   
    fprintf('YOU DID NOT CHOOSE CORRECTLY.\n');
    fprintf('Simulation default: whirlwind\n');

end

%Prints Re Number %
fprintf('\nYour Re is: %d\n\n',Ly*bVel/nu);
fprintf('____________________________________________________________________________\n\n');

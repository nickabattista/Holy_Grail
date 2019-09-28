%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Solves the Navier-Stokes equations in the Velocity-Pressure formulation 
% using a predictor-corrector projection method approach
%
% Author: Nicholas A. Battista
% Created: Novermber 24, 2014
% Modified: September 11, 2019
% 
% Equations of Motion:
% rho Du/Dt = -Nabla(P) + mu*Laplacian(u)  [Conservation of Momentum]
% Nabla \cdot u = 0                        [Conservation of Mass]                                   
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Projection_Method()


% Print key fluid solver ideas to screen
print_Projection_Info();


%
% GRID PARAMETERS %
%
Lx = 1.0;        % Domain Length in x 
Ly = 2.0;        % Domain Length in y 
Nx = 128;        % Spatial Resolution in x 
Ny = 256;        % Spatial Resolution in y
dx = Lx/Nx;      % Spatial Distance Definition in x (NOTE: keep dx = Lx/Nx = Ly/Ny = dy);

%
% SIMULATION PARAMETERS %
%
mu = 1.0;       % Fluid DYNAMIC viscosity (kg / m*s) 
%                    (mu=1000,100,10,1 for Re=4,40,400,4000 respectively for Cavity Flow examples
%                    (mu=0.25,1.0,2.5 for Re=4000,1000, and 400, respectively for Circular Flow examples)
rho = 1000;      % Fluid DENSITY(kg/m^3) 
nu=mu/rho;       % Fluid KINEMATIC viscosity
numPredCorr = 3; % Number of Predictor-Corrector Steps
maxIter=200;     % Maximum Iterations for SOR Method to solve Elliptic Pressure Equation
beta=1.25;       % Relaxation Parameter 

%
% CHOOSE SIMULATION (gives chosen simulation parameters) %
% Possible choices: 'cavity_top', 'whirlwind', 'twoSide_same', 'twoSide_opp', 'corner'
%
choice = 'cavity_top';


%
% Initialize all storage quantities %
%
[u, v, p, uTemp, vTemp, uAvg, vAvg, vorticity, c] = initialize_Storage(Nx,Ny);



%
% Returns Boundary Conditions (BCs) and Other Simulation Parameters for
% specific choice above
%
[uTop,uBot,vRight,vLeft,dt,nStep,pStep,bVel,xStart,yStart] = please_Give_Me_BCs(choice);


%PRINT SIMULATION INFO %
print_Simulation_Info(choice,dt,dx,nu,bVel,Lx,Ly); 


% SAVING INITIAL DATA TO VTK %
print_dump = 200;   % Saves data every print_dump time-steps
ctsave = 0;         % Keeps track of total time-steps
pCount = 0;         % Counter for # of time-steps saved for indexing data
mkdir('vtk_data');  % Create vtk_data FOLDER for .vtk data
print_vtk_files(pCount,u,v,p,vorticity,Lx,Ly,Nx,Ny);



%
% BEGIN TIME-STEPIN'!
% 
t=0.0; %Initialize time
for j=1:nStep
    
    %
    % Enforce Boundary Conditions (Solve for "ghost velocities")
    %        Note: tanh() used to ramp up flow appropriately
    %
    u(1:Nx+1,1)=    (2*uBot-u(1:Nx+1,2) )     * tanh(0.25*t);
    u(1:Nx+1,Ny+2)= (2*uTop-u(1:Nx+1,Ny+1) )  * tanh(0.25*t);
    v(1,1:Ny+1)=    (2*vLeft-v(2,1:Ny+1))     * tanh(0.25*t);
    v(Nx+2,1:Ny+1)= (2*vRight-v(Nx+1,1:Ny+1) )* tanh(0.25*t);

    
    % Start Predictor-Corrector Steps
    for k=1:numPredCorr
    
        %Find auxillary (temporary) velocity fields for predictor step
        [uTemp, vTemp] = give_Auxillary_Velocity_Fields(dt,dx,nu,Nx,Ny,u,v,uTemp,vTemp);
        
        %Solve Elliptic Equation for Pressure via SOR scheme
        p = solve_Elliptic_Pressure_Equation(dt,dx,Nx,Ny,maxIter,beta,c,uTemp,vTemp,p);
        
        % Velocity Correction
        u(2:Nx,2:Ny+1)=uTemp(2:Nx,2:Ny+1)-(dt/dx)*(p(3:Nx+1,2:Ny+1)-p(2:Nx,2:Ny+1));
        v(2:Nx+1,2:Ny)=vTemp(2:Nx+1,2:Ny)-(dt/dx)*(p(2:Nx+1,3:Ny+1)-p(2:Nx+1,2:Ny));
    end
    
    
    %Update Simulation Time 
    t=t+dt;
    
    %
    % UNCOMMENT to PLOT in MATLAB 
    %if ( mod(j,pStep) == 0 )
    %    fprintf('Simulation Time: %d\n',t);
    %    
    %    %Plot Timestep Velocity Fields and Vorticity
    %    plot_Velocity_and_Vorticity(dx,Nx,Ny,xGrid,yGrid,u,v,uAvg,vAvg,vorticity,xStart,yStart);
    %end
    

    % Save files info!
    ctsave = ctsave + 1;
    if mod(ctsave,print_dump) == 0
        
        % increment data storage counter
        pCount = pCount + 1;
        
        % compute vorticity
        vorticity(1:Nx+1,1:Ny+1)=(u(1:Nx+1,2:Ny+2)-u(1:Nx+1,1:Ny+1)-v(2:Nx+2,1:Ny+1)+v(1:Nx+1,1:Ny+1))/(2*dx); 
        
        % call function to store data at this time-step
        print_vtk_files(pCount,u,v,p,vorticity,Lx,Ly,Nx,Ny);
        
        % print simulation time
        fprintf('Simulation Time: %d\n',t);
    end

end %ENDS PROJECTION METHOD TIME-STEPPING


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to initialize storage for velocities, pressure, vorticity,
% coeffs., etc.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [u, v, p, uTemp, vTemp, uAvg, vAvg, vorticity, c] = initialize_Storage(Nx,Ny)

    %Initialize (u,v) velocities, pressure
    u=zeros(Nx+1,Ny+2);      % x-velocity (u) initially zero on grid
    v=zeros(Nx+2,Ny+1);      % y-velocity (v) initially zero on grid
    p=zeros(Nx+2,Ny+2);      % pressure initially zero on grid
    uTemp=zeros(Nx+1,Ny+2);  % auxillary x-Velocity (u*) field initially zero on grid
    vTemp=zeros(Nx+2,Ny+1);  % auxillary y-Velocity (v*) field initially zero on grid

    %Initialize quantities for plotting
    uAvg=zeros(Nx+1,Ny+1);      % uAvg: averaged x-Velocities on grid (smoothing)
    vAvg=zeros(Nx+1,Ny+1);      % vAvg: averaged y-Velocities on grid (smoothing)
    vorticity=zeros(Nx+1,Ny+1); % w: Vorticity

    %Coefficients when solving the Elliptic Pressure Equation w/ SOR (so averaging is consistent)
    c=1/4*ones(Nx+2,Ny+2);   % Interior node coefficients set to 1/4 (all elements exist)
    c(2,3:Ny)=1/3;           % Boundary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c(Nx+1,3:Ny)=1/3;        % Boundary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c(3:Nx,2)=1/3;           % Boundary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c(3:Nx,Ny+1)=1/3;        % Boundary nodes coefficients set to 1/3 (1 element is zero -> nonexistent)
    c(2,2)=1/2;              % Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c(2,Ny+1)=1/2;           % Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c(Nx+1,2)=1/2;           % Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)
    c(Nx+1,Ny+1)=1/2;        % Corner nodes coefficient set to 1/2 (2 elements are zero -> nonexistent in computation)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to find auxillary (temporary) velocity fields in predictor step
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [uTemp, vTemp] = give_Auxillary_Velocity_Fields(dt,dx,nu,Nx,Ny,u,v,uTemp,vTemp)
    

    %Find Temporary u-Velocity Field
    for i=2:Nx 
        for j=2:Ny+1 
            uTemp(i,j)=u(i,j)+dt*(-(0.25/dx)*((u(i+1,j)+u(i,j))^2-(u(i,j)+u(i-1,j))^2+(u(i,j+1)+u(i,j))*(v(i+1,j)+v(i,j))-(u(i,j)+u(i,j-1))*(v(i+1,j-1)+v(i,j-1)))+(nu/dx^2)*(u(i+1,j)+u(i-1,j)+u(i,j+1)+u(i,j-1)-4*u(i,j)));
        end
    end

    %Find Temporary v-Velocity Field
    for i=2:Nx+1
        for j=2:Ny 
            vTemp(i,j)=v(i,j)+dt*(-(0.25/dx)*((u(i,j+1)+u(i,j))*(v(i+1,j)+v(i,j))-(u(i-1,j+1)+u(i-1,j))*(v(i,j)+v(i-1,j))+(v(i,j+1)+v(i,j))^2-(v(i,j)+v(i,j-1))^2)+(nu/dx^2)*(v(i+1,j)+v(i-1,j)+v(i,j+1)+v(i,j-1)-4*v(i,j)));
        end
    end

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to solve elliptic pressure equation using a SOR scheme 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = solve_Elliptic_Pressure_Equation(dt,dx,Nx,Ny,maxIter,beta,c,uTemp,vTemp,p)

    iter = 1; err = 1; tol = 5e-6;
    pPrev = p;
    while ( (err > tol) && (iter < maxIter) )     
        for i=2:Nx+1 
            for j=2:Ny+1
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
    
function plot_Velocity_and_Vorticity(dx,Nx,Ny,xGrid,yGrid,u,v,uAvg,vAvg,vorticity,xStart,yStart)

    %Plot Info
    xF = xGrid(1);   %Left most x-Point
    xL = xGrid(end); %Right most x-Point
    yF = yGrid(1);   %Bottom most y-Point
    yL = yGrid(end); %Top most y-Point
    
    %Compute x and y directed velocity averages (interpolate from cell-staggered to grid points)
    uAvg(1:Nx+1,1:Ny+1)=0.5*(u(1:Nx+1,2:Ny+2)+u(1:Nx+1,1:Ny+1));
    vAvg(1:Nx+1,1:Ny+1)=0.5*(v(2:Nx+2,1:Ny+1)+v(1:Nx+1,1:Ny+1));

    %Compute magnitude of velocity
    velMag = sqrt( uAvg.*uAvg + vAvg.*vAvg );
    
    %Compute Vorticity
    vorticity(1:Nx+1,1:Ny+1)=(u(1:Nx+1,2:Ny+2)-u(1:Nx+1,1:Ny+1)-v(2:Nx+2,1:Ny+1)+v(1:Nx+1,1:Ny+1))/(2*dx); 
    
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
%          NOTE: all velocities normal to the boundary are zero.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function [uTop,uBot,vRight,vLeft,dt,nStep,printStep,bVel,xStart,yStart] = please_Give_Me_BCs(choice)

% Possible choices: 'cavity_top', 'whirlwind', 'twoSide_same', 'twoSide_opp', 'corner'


if strcmp(choice,'cavity_top')
    
    bVel = 4.0;
    uTop = bVel; uBot = 0.0; vRight = 0.0; vLeft = 0;

    endTime = 6.0;                % Final time in simulation
    dt = 0.001;                   % Time-step (5e-5 for Re4, 1e-4 for Re40, 1e-3 for Re400+Re4000)
    nStep=floor(endTime/dt);      % Number of Time-Steps
    printStep = 10;               % Plot data (MATLAB) every # of printStep frames
    
    % Streamlines Info (for MATLAB plotting) %
    xStart = [0.1 0.5 0.7];
    yStart = 0.5*ones(size(xStart));
        
elseif strcmp(choice,'whirlwind')
    
    bVel = 1.0;
    uTop=bVel;  uBot=-bVel; vRight=-bVel; vLeft=bVel;
    
    endTime = 24;            % Final time in simulation
    dt = 0.001;              % Time-step
    nStep=floor(endTime/dt); % Number of Time-Steps
    printStep = 10;          % Plot data (MATLAB) every # of printStep frames
    
    % Streamlines Info (for MATLAB plotting) %
    yStart = 0.10:0.15:0.40;
    xStart = ones(size(yStart));
    
elseif strcmp(choice,'twoSide_same')
    
    bVel = 2.0;
    uTop = 0.0; uBot = 0.0; vRight = bVel; vLeft = bVel;
    
    dt = 0.01;      % Time-step
    nStep=300;      % Number of Time-Steps
    printStep = 3;  % Plot data (MATLAB) every # of printStep frames
    
    % Streamlines Info (for MATLAB plotting) %
    xStart = [0.1 0.5 1.55 1.9];
    yStart = 0.5*ones(size(xStart));

    
elseif strcmp(choice,'twoSide_opp')
    
    bVel = 2.0;
    uTop = 0.0; uBot = 0.0; vRight = -bVel; vLeft = bVel;
    
    dt = 0.01;      % Time-step
    nStep=300;      % Number of Time-Steps
    printStep = 3;  % Plot data (MATLAB) every # of printStep frames
    
    % Streamlines Info (for MATLAB plotting) %
    xStart = [0.1 0.5 0.9 1.1 1.55 1.9];
    yStart = 0.5*ones(size(xStart));
    
elseif strcmp(choice,'corner')

    bVel = 1.0;
    uTop = bVel; uBot = 0.0; vRight = 0; vLeft = bVel;
    
    dt = 0.01;      % Time-step
    nStep=300;      % Number of Time-Steps
    printStep = 3;  % Plot data (MATLAB) every # of printStep frames
    
    % Streamlines Info (for MATLAB plotting) %
    yStart = 0.10:0.15:0.55;
    xStart = ones(size(yStart));
    
else
   
    fprintf('YOU DID NOT CHOOSE CORRECTLY!!!!!\n');
    fprintf('Simulation DEFAULT: whirlwind\n');
    
    bVel = 1.0;
    uTop=bVel;  uBot=-bVel; vRight=-bVel; vLeft=bVel;
    
    endTime = 24;            % Final time in simulation
    dt = 0.001;              % Time-step
    nStep=floor(endTime/dt); % Number of Time-Steps
    printStep = 10;          % Plot data (MATLAB) every # of printStep frames
    
    % Streamlines Info (for MATLAB plotting) %
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
fprintf('Modified: September 11, 2019\n');
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

function print_Simulation_Info(choice,dt,dx,nu,bVel,Lx,Ly)

% PRINT STABILITY INFO %
fprintf('\nNOTE: dt must be <= %d for any chance of STABILITY!\n',0.25*dx^2/nu);
fprintf('Your dt = %d\n\n',dt);

if strcmp(choice,'cavity_top')
    
    fprintf('You are simulating cavity flow\n');
    fprintf('The open cavity is on the TOP side\n');
    fprintf('Try changing the viscosity or geometry\n\n');
    
elseif strcmp(choice,'whirlwind')
    
    fprintf('You are simulating vortical flow\n');
    fprintf('All velocities on the wall point at the next corner in a CW manner\n');
    fprintf('Try changing the velocity BCs on each wall\n');
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
fprintf('\nYour Re is: %d\n\n',Lx*bVel/nu);
fprintf('____________________________________________________________________________\n\n');
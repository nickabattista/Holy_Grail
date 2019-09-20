%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 2D LATTICE BOLTZMANN (LBM) SIMULATION 
% Author: Nicholas A. Battista
% Created: 11/4/2014  (MATLAB)
% Created: 5/5/2017   (Python3)
% Modified: September 13, 2019 (MATLAB, Python)
%
%  D2Q9 Model:
%  c6  c2   c5  
%    \  |  /    
%  c3- c9 - c1  
%    /  |  \   
%  c7  c4   c8     
%
% f_i: the probability for site vec(x) to have a particle heading in
%     direction i, at time t. These called discretized probability
%     distribution functions and represent the central link to LBMs.
%
% LBM Idea: 
%         1. At each timestep the particle densities propogate in each direction (1-8). 
%         2. An equivalent "equilibrium' density is found
%         3. Densities relax towards that state, in proportion governed by
%            tau (parameter related to viscosity).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function lets_do_LBM()


%Prints key ideas to screen
print_LBM_Info(); 


%
% Simulation Parameters
%
tau=0.53;                    % tau: relaxation parameter related to viscosity
density=0.01;                % density to be used for initializing whole grid to value 1.0
w1=4/9; w2=1/9; w3=1/36;      % weights for finding equilibrium distribution
Nx=640; Ny=160;               % number of grid cells in x and y directions, respectively
Lx = 2; Ly = 0.5;             % Size of computational domain
dx = Lx/Nx; dy = Ly/Ny;       % Grid Resolution in x and y directions, respectively


%
% Gridding Initialization
%
f=repmat(density/9,[Nx Ny 9]);% Copies density/9 into 9-matrices of size [Nx,Ny] -> ALLOCATION for all "DIRECTIONS"
f_EQ = f;                     % Initializes F-equilibrium Storage space
grid_size=Nx*Ny;              % Total number of grid cells
CI= 0:grid_size:7*grid_size;  % Indices to point to FIRST entry of the desired "z-stack" distribution grid      


%
% Chooses which problem to simulate
%
% Possible Choices: 'cylinder1', 'cylinder2', 'channel', 'porous1','porous2','porousCylinder'
%
choice = 'porousCylinder';
percentPorosity = 0.625;  % Percent of Domain that's Porous (does not matter if not studying porous problem)
[BOUND,Bound2,deltaU,endTime] = give_Me_Problem_Geometry(choice,Nx,Ny,percentPorosity); %BOUND: gives geometry, deltaU: gives incremental increase to inlet velocity
print_simulation_info(choice);



%Find Indices of NONZERO Elements, i.e., where "boundary points" IS
ON=find(BOUND');      %matrix index of each Occupied Node (need transpose due to (x,y) matrix convention, e.g., row-id = x, col-id = y)
ON_geo = find(Bound2); %matrix index of each Occupied Node

% Give Boundary Points For Saving Data
Bound_Pts = give_Me_Boundary_Pts_For_Visualization(dx,dy,Nx,Ny,Lx,Ly,ON_geo);


%Offsets Indices for the Different Directions [i.e., levels of F_i=F(:,:,i) ] for known BOUNDARY pts.
TO_REFLECT=[ON+CI(1) ON+CI(2) ON+CI(3) ON+CI(4) ON+CI(5) ON+CI(6) ON+CI(7) ON+CI(8)];
REFLECTED= [ON+CI(3) ON+CI(4) ON+CI(1) ON+CI(2) ON+CI(7) ON+CI(8) ON+CI(5) ON+CI(6)];


%Initialization Parameters
ts=0;                             %initialize starting time to 0 (time-step)
fprintf('Simulation Time: %d\n',ts);

% SAVING DATA TO VTK %
print_dump = 400;%floor(endTime/50);
ctsave = 0; % Counts # of total time-steps
pSave = 0;  % Counts # of time-steps with data saved

% CREATE VIZ_IB2D FOLDER and VISIT FILES
mkdir('vtk_data');
UX = zeros(Nx,Ny); UY = UX; vorticity = UX(1:end-1,1:end-1);
print_vtk_files(pSave,UX,UY,vorticity,Lx,Ly,Nx,Ny,Bound_Pts);



%Begin time-stepping!
while ts < endTime
    
    
    % STREAMING STEP (progate in respective directions)
    f = please_Stream_Distribution(f,Nx,Ny);
        
    
    %Densities bouncing back at next timestep
    BOUNCEDBACK=f(TO_REFLECT);
    

    %vec(rho) = SUM_i f_i -> SUMS EACH DISTRIBUTION MATRIX TOGETHER
    DENSITY=sum(f,3);   %Note: '3' denotes sum over third dimension
    
    %vec(u) = 1/vec(rho) SUM_i (f_i)(e_i) -> CREATES VELOCITY MATRICES
    UX=( sum(f(:,:,[1 5 8]),3)-sum(f(:,:,[3 6 7]),3) ) ./ DENSITY; 
    UY=( sum(f(:,:,[2 5 6]),3)-sum(f(:,:,[4 7 8]),3) ) ./ DENSITY;
    
    
    %Increase inlet velocity with each time step along left wall
    % flow past cylinders
    UX(1,2:Ny-1) = UX(1,2:Ny-1) + deltaU;

    % porous
    %UX(1,floor(1/3*Ny):ceil(2/3*Ny)) = UX(1,floor(1/3*Ny):ceil(2/3*Ny)) + deltaU;
    
    %Enforce BCs to Zero Velocity / Zero Density
    UX(ON)=0;      %Makes all Boundary Regions have zero x-velocity 
    UY(ON)=0;      %Makes all Boundary Regions have zero y-velocity
    DENSITY(ON)=0; %Makes DENSITY of Boundary Regions have zero value.
    
    
    %Square of Magnitude of Velocity Overall
    U_SQU = UX.^2 + UY.^2; 
    
    %Create "Diagonal" Velocity Quantities
    U_5 =  UX+UY; %Create velocity direction to Point 5
    U_6 = -UX+UY; %Create velocity direction to Point 6
    U_7 = -U_5;   %Create velocity direction to Point 7
    U_8 = -U_6;   %Create velocity direction to Point 8
    
    %Calculate the equilibrium distribution
    f_EQ = please_Give_Equilibrium_Distribution(w1,w2,w3,DENSITY,UX,UY,U_SQU,U_5,U_6,U_7,U_8,f_EQ);
        
    %Update the PDFs
    f = f - (1/tau)*(f-f_EQ);
    
    %BOUNCE BACK DENSITIES for next time-step
    f(REFLECTED)= BOUNCEDBACK;
    
    %Updates simulation parameters
    ts=ts+1;   %update time step
    
    % Save files info!
    ctsave = ctsave + 1;
    if mod(ctsave,print_dump) == 0

        % increment pSave
        pSave = pSave + 1;
        
        % compute vorticity
        dUx_y = ( UX(1:Nx-1,2:Ny)-UX(1:Nx-1,1:Ny-1) ) / dy;
        dUy_x = ( UY(2:Nx,1:Ny-1)-UY(1:Nx-1,1:Ny-1) ) / dx;
        vorticity(1:Nx-1,1:Ny-1)=( dUy_x - dUx_y );
        
        % print to vtk
        print_vtk_files(pSave,UX,UY,vorticity,Lx,Ly,Nx,Ny,Bound_Pts);
        fprintf('Simulation Time: %d\n',ts);
            
    end
    
end

%Plots the "steady-state" velocities
plot_Steady_State(UX,UY,BOUND,Nx,Ny,ts,choice);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to choose what geometry to consider for the simulation
% Returns: Geometry / Increase to inlet velocity for each time-step / endTime
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [BOUND, Bound2, deltaU, endTime] = give_Me_Problem_Geometry(choice,Nx,Ny,percentPorosity)


if strcmp(choice,'cylinder1')

    %
    % FLOW PAST CYLINDER EXAMPLE
    % 
    % WORKS WELL: <faster, less accurate> [Nx,Ny]=[128,512], tau=0.53, density = 0.01, endTime = 5500
    %             <slower, more accurate> [Nx,Ny]=[256,1024],tau=0.55, density = 0.01
    
    % radius of cylinder (centered in middle of domain, given in terms of mesh widths)
    r = 0.075*Ny;
    
    % Creates (x,y)-proxy to define geometry upon (based on 1:1:Nx, 1:1:Ny)
    a1=repmat(-(Nx-1)/2:(Nx-1)/2,[Ny,1]); 
    a2=repmat(-(Ny-1)/2:(Ny-1)/2,[Nx,1]);
    
    % Amount to translate cylinder from middle of domain
    aR = floor(0.375*Nx);                
    
    % CREATE CYLINDER GEOMETRY (Note: "+" shifts left bc of defining circles)
    BOUND=( (a1+aR).^2+(a2)'.^2)<r^2;   % Puts "1's" within region of Cylinder 
    
    % CREATE TOP/BOTTOM BOUNDARIES
    BOUND([1 Ny],1:Nx)=1;               % Puts "1's" on Top/Bottom Boundaries
    
    % Simulation characteristics
    deltaU = 0.00125;                     % Incremental increase to inlet velocity
    endTime = 56000;                      % Total Number of Time-Steps
    
elseif strcmp(choice,'cylinder2')

    %
    % FLOW PAST MULTIPLE CYLINDERs EXAMPLE
    % 
    % WORKS WELL: <faster, less accurate> [Nx,Ny]=[128,512], tau=0.53, density = 0.01, endTime = 5500
    %             <slower, more accurate> [Nx,Ny]=[256,1024],tau=0.55, density = 0.01, endTime = 100000

    
    % radius of bubble (centered in middle of domain, given in terms of mesh widths)
    r = 0.125*Ny;
    
    % Creates (x,y)-proxy to define geometry upon (based on 1:1:Nx, 1:1:Ny)
    a1=repmat(-(Nx-1)/2:(Nx-1)/2,[Ny,1]); 
    a2=repmat(-(Ny-1)/2:(Ny-1)/2,[Nx,1]);
    
    % Amount to translate cylinder from middle of domain
    aR = floor(0.375*Nx);    
    aSx = floor(0.1*Nx);
    aY = floor(0.2*Ny);
    
    % CREATE CYLINDER GEOMETRY (Note: "+" shifts left bc of defining circles)
    B1 =( (a1+aR).^2+(a2)'.^2)<r^2;   % Puts "1's" within region of Cylinder  
    B2 =( (a1+aR-aSx).^2+(a2-aY)'.^2)<r^2;   % Puts "1's" within region of Cylinder  
    B3 =( (a1+aR-aSx).^2+(a2+aY)'.^2)<r^2;   % Puts "1's" within region of Cylinder  

    % COMBINE together all cylinder geometry information
    BOUND = double(B1)+double(B2)+double(B3);          
   
    % CREATE TOP/BOTTOM BOUNDARIES
    BOUND([1 Ny],1:Nx)=1;               % Puts "1's" on Top/Bottom Boundaries    
    
    % Simulation characteristics
    deltaU = 0.01;                                    % Incremental increase to inlet velocity
    endTime = 5500;                                   % Total Number of Time-Steps

    size(BOUND)
    pause();
    
elseif strcmp(choice,'channel')
    
    %CHANNEL GEOMETRY
    BOUND=zeros(Ny,Nx);
    BOUND([1 Ny],1:Nx)=1;               % Puts "1's" on Top/Bottom Boundaries    
    deltaU = 0.01;                      % Incremental increase to inlet velocity
    endTime = 2500;                     % Total Number of Time-Steps


elseif strcmp(choice,'porous1')
    
    %POROUS RANDOM DOMAIN
    BOUND=rand(Ny,Nx)<1-percentPorosity;   %PUTS "1's" inside domain randomly if RAND value above percent  
    aS = ceil(2/5*Nx);
    aE = ceil(3/5*Nx);
    BOUND(:,1:aS) = 0; 
    BOUND(:,aE:end)=0;
    BOUND([1 Ny],1:Nx)=0;                 % Puts "1's" on Top/Bottom Boundaries    
    deltaU = 1e-7;                        % Incremental increase to inlet velocity
    endTime = 50000;                      % Total Number of Time-Steps
    
elseif strcmp(choice,'porous2')
    
    %POROUS RANDOM DOMAIN
    BOUND=rand(Ny,Nx)<1-percentPorosity;  % PUTS "1's" inside domain randomly if RAND value above percent              
    BOUND(:,1:floor(9*Nx/31)) = 0;                   % PUTS "0's" to make open channels through porous structure
    BOUND(:,floor(7*Nx/31):floor(9*Nx/31)) = 0;                   % PUTS "0's" to make open channels through porous structure
    BOUND(:,floor(13*Nx/31):floor(15*Nx/31)) = 0;                 % PUTS "0's" to make open channels through porous structure
    BOUND(:,floor(19/31*Nx):floor(21/31*Nx)) = 0;                 % PUTS "0's" to make open channels through porous structure
    BOUND(:,floor(25/31*Nx):floor(27/31*Nx))=0;                   % PUTS "0's" to make open channels through porous structure
    BOUND(:,floor(30/31*Nx):end) = 0;                % PUTS "0's" to make open channels through porous structure
    BOUND([1 Ny],1:Nx)=0;               % Puts "1's" on Top/Bottom Boundaries    
    deltaU = 1e-7;                      % Incremental increase to inlet velocity
    endTime = 50000;                     % Total Number of Time-Steps

elseif strcmp(choice,'porousCylinder')

    %
    % FLOW PAST CYLINDER EXAMPLE
    % 
    % WORKS WELL: <faster, less accurate> [Nx,Ny]=[128,512], tau=0.53, density = 0.01, endTime = 5500
    %             <slower, more accurate> [Nx,Ny]=[256,1024],tau=0.55, density = 0.01
    
    % radius of cylinder (centered in middle of domain, given in terms of mesh widths)
    r = 0.075*Ny;
    
    % Creates (x,y)-proxy to define geometry upon (based on 1:1:Nx, 1:1:Ny)
    a1=repmat(-(Nx-1)/2:(Nx-1)/2,[Ny,1]); 
    a2=repmat(-(Ny-1)/2:(Ny-1)/2,[Nx,1]);
    
    % Amount to translate cylinder from middle of domain
    aR = floor(0.375*Nx);                
    
    % CREATE CYLINDER GEOMETRY (Note: "+" shifts left bc of defining circles)
    BOUND=( (a1+aR).^2+(a2)'.^2)<r^2;   % Puts "1's" within region of Cylinder 
    
    % FIND 1's
    indVec = find(BOUND==1);
    
    % CONVERT BOOLEAN MATRIX INTO MATRIX OF DOUBLE VALUES
    BOUND = double(BOUND);
    
    % CHANGE 1's to RANDOM NUMBERS
    BOUND(indVec) = rand( size(indVec) );

    % MAKE POROSITY STATEMENT
    BOUND = ( BOUND > percentPorosity );  % PUTS "1's" inside domain randomly if RAND value above percent
    
    % CREATE TOP/BOTTOM BOUNDARIES
    BOUND([1 Ny],1:Nx)=1;               % Puts "1's" on Top/Bottom Boundaries
    
    % Simulation characteristics
    deltaU = 0.00125;                     % Incremental increase to inlet velocity
    endTime = 56000;                      % Total Number of Time-Steps    
    
end


%
% Reverse Order of BOUND for GEOMETRY (top is initialized as bottom and vice versa)
%
Bound2=zeros( size(BOUND) );
for i=0:Ny-1
    Bound2(i+1,:) = BOUND(Ny-i,:);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to stream the distribution function, f.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = please_Stream_Distribution(f,Nx,Ny)
   

f(:,:,1)=f([Nx 1:Nx-1],:,1);          %Stream Right

f(:,:,2)=f(:,[Ny 1:Ny-1],2);          %Stream Up

f(:,:,3)=f([2:Nx 1],:,3);             %Stream Left

f(:,:,4)=f(:,[2:Ny 1],4);             %Stream Down

f(:,:,5)=f([Nx 1:Nx-1],[Ny 1:Ny-1],5);%Stream Right-Up

f(:,:,6)=f([2:Nx 1],[Ny 1:Ny-1],6);   %Stream Left-Up

f(:,:,7)=f([2:Nx 1],[2:Ny 1],7);      %Stream Left-Down    

f(:,:,8)=f([Nx 1:Nx-1],[2:Ny 1],8);   %Stream Right-Down


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to give the equilibrium distribution, f_EQ.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f_EQ = please_Give_Equilibrium_Distribution(w1,w2,w3,DENSITY,UX,UY,U_SQU,U_5,U_6,U_7,U_8,f_EQ)
     
    % Calculate equilibrium distribution: stationary pt in middle.
    f_EQ(:,:,9)= w1*DENSITY .* (1 - (3/2)*U_SQU );
    
   
    % NEAREST-neighbours (i.e., stencil pts directly right,left,top,bottom)
    % Equilibrium DFs can be obtained from the local Maxwell-Boltzmann SPDF 
    f_EQ(:,:,1)=w2*DENSITY .* (1 + 3*UX + (9/2)*(UX).^2 - (3/2)*U_SQU );
    f_EQ(:,:,2)=w2*DENSITY .* (1 + 3*UY + (9/2)*(UY).^2 - (3/2)*U_SQU );
    f_EQ(:,:,3)=w2*DENSITY .* (1 - 3*UX + (9/2)*(UX).^2 - (3/2)*U_SQU );
    f_EQ(:,:,4)=w2*DENSITY .* (1 - 3*UY + (9/2)*(UY).^2 - (3/2)*U_SQU );
    
    % NEXT-NEAREST neighbours (i.e., diagonal elements for stencil pts)
    % Equilibrium DFs can be obtained from the local Maxwell-Boltzmann SPDF 
    f_EQ(:,:,5)=w3*DENSITY .* (1 + 3*U_5 + (9/2)*(U_5).^2 - (3/2)*U_SQU );
    f_EQ(:,:,6)=w3*DENSITY .* (1 + 3*U_6 + (9/2)*(U_6).^2 - (3/2)*U_SQU );
    f_EQ(:,:,7)=w3*DENSITY .* (1 + 3*U_7 + (9/2)*(U_7).^2 - (3/2)*U_SQU );
    f_EQ(:,:,8)=w3*DENSITY .* (1 + 3*U_8 + (9/2)*(U_8).^2 - (3/2)*U_SQU );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to give boundary points for visualization in vtk printing.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function Bound_Pts = give_Me_Boundary_Pts_For_Visualization(dx,dy,Nx,Ny,Lx,Ly,ON)


    % Initialize xMat & vMat matrices
    xMat = zeros(Ny,Nx);
    yMat = zeros(Ny,Nx);

    % Construct Matrix of x-Values
    xVec = 0:dx:Lx-dx;
    for i=1:Ny
       xMat(i,:) = xVec; 
    end
    
    % Construct Matrix of y-Values
    yVec = (Ly-dy:-dy:0)';
    for i=1:Nx
        yMat(:,i) = yVec;
    end
    
    % Get x-Val and y-Val of each boundary point
    xBounds = xMat(ON);
    yBounds = yMat(ON);
    Bound_Pts = zeros(length(xBounds),2);
    for i =1:length(xBounds)
        Bound_Pts(i,1) = xBounds(i);
        Bound_Pts(i,2) = yBounds(i);
    end
    
    %subplot(1,2,1)
    %image(2-BOUND'); hold on;
    %subplot(1,2,2)
    %plot(Bound_Pts(:,1),Bound_Pts(:,2),'*');
    %pause();    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to PLOT the steady-state velocities, UX and UY, at time-step, ts.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_Steady_State(UX,UY,BOUND,Nx,Ny,ts,choice)


%Set up grids
xGrid = 1:1:Nx;
yGrid = 1:1:Ny;

%Scale Factor for Velocity Vectors
scale = 5;

%Streamlines
yStart = floor(Nx/3):2:floor(2*Nx/3);
xStart = 2*ones(size(yStart));
options = [0.01 10000];

%Compute magnitude of velocity
velMag = sqrt( UX.*UX + UY.*UY );

%Compute vorticity (approximate vorticity = dUy/dx - dYx/dy )
vorticity = (UY - UX)/2;
dUx_y = UX(1:Nx-1,2:Ny)-UX(1:Nx-1,1:Ny-1);
dUy_x = UY(2:Nx,1:Ny-1)-UY(1:Nx-1,1:Ny-1);
vorticity(1:Nx-1,1:Ny-1)=( dUy_x - dUx_y )/(2); 

% Quiver-plot has error in Nx ~= Ny
if Nx==Ny

    if strcmp(choice,'channel')

        figure(1);
        subplot(1,2,1)
        colormap(gray(2));
        image(2-BOUND');hold on;
        quiver(xGrid,yGrid,UX(xGrid,:)',UY(yGrid,:)');
        title(['Velocity Field at',num2str(ts),'\Deltat']);
        xlabel('x');
        ylabel('y');

        xPts = [floor(Nx/4) floor(2*Nx/4) floor(3*Nx/4)];
        yVec = 1:Ny;
        uMag1 = sqrt( UX(xPts(1),:).*UX(xPts(1),:) + UY(xPts(1),:).*UY(xPts(1),:)  );
        uMag2 = sqrt( UX(xPts(2),:).*UX(xPts(2),:) + UY(xPts(2),:).*UY(xPts(2),:)  );
        uMag3 = sqrt( UX(xPts(3),:).*UX(xPts(3),:) + UY(xPts(3),:).*UY(xPts(3),:)  );
        mat(1,:) = uMag1;
        mat(2,:) = uMag2;
        mat(3,:) = uMag3;

        subplot(1,2,2)
        plot(yVec,uMag1,'.-','MarkerSize',7); hold on;
        plot(yVec,uMag2,'r.-','MarkerSize',7); hold on;
        plot(yVec,uMag3,'g.-','MarkerSize',7); hold on;
        xlabel('y');
        ylabel('Velocity Magnitude'); 
        title('Cross-sectional velocities along tube');
        legend('1/3 into tube','2/3 into tube','3/3 into tube');
        maxy = 1.05*max(max(mat));
        axis([0 Nx 0 maxy]);

    elseif ( strcmp(choice,'cylinder1') || strcmp(choice,'cylinder2') )

        figure(1);
        subplot(1,2,1)
        colormap(gray(2));
        image(2-BOUND');hold on;
        quiver(xGrid,yGrid,UX(xGrid,:)',UY(yGrid,:)');
        title(['Velocity Field at',num2str(ts),'\Deltat']);
        xlabel('x');
        ylabel('y');

        xPts = [floor(Nx/5) floor(2*Nx/5) floor(3*Nx/5) floor(4*Nx/5)];
        yVec = 1:Ny;
        uMag1 = sqrt( UX(xPts(1),:).*UX(xPts(1),:) + UY(xPts(1),:).*UY(xPts(1),:)  );
        uMag2 = sqrt( UX(xPts(2),:).*UX(xPts(2),:) + UY(xPts(2),:).*UY(xPts(2),:)  );
        uMag3 = sqrt( UX(xPts(3),:).*UX(xPts(3),:) + UY(xPts(3),:).*UY(xPts(3),:)  );
        uMag4 = sqrt( UX(xPts(4),:).*UX(xPts(4),:) + UY(xPts(4),:).*UY(xPts(4),:)  );
        mat(1,:) = uMag1;
        mat(2,:) = uMag2;
        mat(3,:) = uMag3;
        mat(4,:) = uMag4;

        subplot(1,2,2)
        plot(yVec,uMag1,'.-','MarkerSize',7); hold on;
        plot(yVec,uMag2,'k.-','MarkerSize',7); hold on;
        plot(yVec,uMag3,'r.-','MarkerSize',7); hold on;
        plot(yVec,uMag4,'g.-','MarkerSize',7); hold on;
        xlabel('y');
        ylabel('Velocity Magnitude'); 
        title('Cross-sectional Velocities Along Channel');
        legend('Radii before','Middle cylinder','Radii after','3/4 down tube');
        maxy = 1.05*max(max(mat));
        axis([0 Nx 0 maxy]);

    elseif strcmp(choice,'porous1')

        figure(1);
        subplot(1,2,1);
        colormap(gray(2));
        uMag = sqrt( UX.*UX + UY.*UY );
        scaley = 5.5/max( max( uMag ));
        image(2-BOUND');hold on;
        quiver(xGrid(2:end),yGrid(1:end),scaley*UX(xGrid(2:end),:)',scaley*UY(yGrid(2:end),:)','AutoScale','off'); hold on;
        title(['Velocity Field at ',num2str(ts),'\Deltat']);
        xlabel('x');
        ylabel('y');
        axis([1 Nx 1 Ny]);

        xPts = [floor(Nx/15) floor(Nx/5) Nx-floor(Nx/5) Nx-floor(Nx/15)];
        %xPts = [2 6 26 31];
        yVec = 1:Ny;
        uMag1 = sqrt( UX(xPts(1),:).*UX(xPts(1),:) + UY(xPts(1),:).*UY(xPts(1),:)  );
        uMag2 = sqrt( UX(xPts(2),:).*UX(xPts(2),:) + UY(xPts(2),:).*UY(xPts(2),:)  );
        uMag3 = sqrt( UX(xPts(3),:).*UX(xPts(3),:) + UY(xPts(3),:).*UY(xPts(3),:)  );
        uMag4 = sqrt( UX(xPts(4),:).*UX(xPts(4),:) + UY(xPts(4),:).*UY(xPts(4),:)  );
        mat(1,:) = uMag1;
        mat(2,:) = uMag2;
        mat(3,:) = uMag3;
        mat(4,:) = uMag4;

        subplot(1,2,2)
        plot(yVec,uMag1,'.-','MarkerSize',7); hold on;
        plot(yVec,uMag2,'k.-','MarkerSize',7); hold on;
        plot(yVec,uMag3,'r.-','MarkerSize',7); hold on;
        plot(yVec,uMag4,'g.-','MarkerSize',7); hold on;
        xlabel('y');
        ylabel('Velocity Magnitude'); 
        title('Cross-sectional Velocities Along Channel');
        legend('Inflow','Right before','Right after','Outflow');
        maxy = 1.05*max(max(mat));
        axis([0 Nx 0 maxy]);

    elseif strcmp(choice,'porous2')

        figure(1);
        subplot(1,2,1);
        colormap(gray(2));
        uMag = sqrt( UX.*UX + UY.*UY );
        scaley = 5.5/max( max( uMag ));
        image(2-BOUND');hold on;
        quiver(xGrid(2:end),yGrid(1:end),scaley*UX(xGrid(2:end),:)',scaley*UY(yGrid(2:end),:)','AutoScale','off'); hold on;
        title(['Velocity Field at ',num2str(ts),'\Deltat']);
        xlabel('x');
        ylabel('y');
        axis([1 Nx 1 Ny]);

        xPts = [floor(Nx/5) floor(2*Nx/5) floor(3*Nx/5) floor(4*Nx/5)];
        %xPts = [8 14 20 26];
        yVec = 1:Ny;
        uMag1 = sqrt( UX(xPts(1),:).*UX(xPts(1),:) + UY(xPts(1),:).*UY(xPts(1),:)  );
        uMag2 = sqrt( UX(xPts(2),:).*UX(xPts(2),:) + UY(xPts(2),:).*UY(xPts(2),:)  );
        uMag3 = sqrt( UX(xPts(3),:).*UX(xPts(3),:) + UY(xPts(3),:).*UY(xPts(3),:)  );
        uMag4 = sqrt( UX(xPts(4),:).*UX(xPts(4),:) + UY(xPts(4),:).*UY(xPts(4),:)  );
        mat(1,:) = uMag1;
        mat(2,:) = uMag2;
        mat(3,:) = uMag3;
        mat(4,:) = uMag4;

        subplot(1,2,2)
        plot(yVec,uMag1,'.-', 'MarkerSize',7); hold on;
        plot(yVec,uMag2,'k.-','MarkerSize',7); hold on;
        plot(yVec,uMag3,'r.-','MarkerSize',7); hold on;
        plot(yVec,uMag4,'g.-','MarkerSize',7); hold on;
        xlabel('y');
        ylabel('Velocity Magnitude'); 
        title('Cross-sectional Velocities Along Channel');
        legend('After 1st','After 2nd','After 3rd','After 4th');
        maxy = 1.05*max(max(mat));
        axis([0 Nx 0 maxy]);

    end

end

figure(2)
subplot(1,2,1)
    hold off
    contourf(xGrid,yGrid,velMag',7);hold on;
    quiver(xGrid,yGrid,scale*UX(1:Nx,:)',scale*UY(1:Nx,:)','k','AutoScale','off');hold on;
    title('Velocity Magnitude Colormap');
    axis equal
    axis([1 Nx 1 Ny]);

subplot(1,2,2)
    hold off
    contourf(xGrid,yGrid,vorticity',3); hold on;
    h = streamline(xGrid,yGrid,scale*UX(1:Nx,:)',scale*UY(1:Nx,:)',xStart,yStart,options); hold on;
    set(h,'linewidth',1.25,'color','k');
    plot(xStart,yStart,'r.','MarkerSize',6); hold on;
    title('Vorticity Colormap and Streamlines');
    axis equal
    axis([1 Nx 1 Ny]);
    



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to print Lattice Boltzmann key ideas to screen.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_LBM_Info()

fprintf('\n\n 2D LATTICE BOLTZMANN (LBM) SIMULATION \n');
fprintf('Author: Nicholas A. Battista\n');
fprintf('Created: 11/4/2014  (MATLAB)\n');
fprintf('Modified: 12/2/2014 (MATLAB)\n');
fprintf('Created: 5/5/2017   (Python3)\n\n');
fprintf('_____________________________________________________________________________\n\n');
fprintf('D2Q9 Model:\n\n');
fprintf('c6  c2   c5\n');
fprintf('  \\  |  /  \n');
fprintf('c3- c9 - c1\n');
fprintf('  /  |  \\  \n');
fprintf('c7  c4   c8\n\n');

fprintf('f_i: the probability for site vec(x) to have a particle heading in\n');
fprintf('direction i, at time t. These f_i''s are called discretized probability \n');
fprintf('distribution functions\n\n');

fprintf('LBM Idea: \n');
fprintf('1. At each timestep the particle densities propogate in each direction (1-8).\n');
fprintf('2. An equivalent "equilibrium" density is found\n');
fprintf('3. Densities relax towards that state, in proportion governed by tau\n');
fprintf('%s (parameter related to viscosity)\n\n','  ');
fprintf('_____________________________________________________________________________\n\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to print specific simulation info to screen
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_simulation_info(choice)

if strcmp(choice,'channel')
    
    fprintf('You are simulating CHANNEL FLOW\n');
    fprintf('Flow proceeds left to right through the channel\n');
    fprintf('You should see a parabolic flow profile develop\n\n\n');
    
elseif strcmp(choice,'cylinder1')
    
    fprintf('You are simulating flow around a cylinder\n');
    fprintf('Flow proceeds left to right through the channel containing a 2D cylinder\n');
    fprintf('You should see flow wrapping around the cylinder\n');
    fprintf('Try changing the tau (viscosity) to observe differing dynamics\n\n\n');
    
elseif strcmp(choice,'cylinder2')
    
    fprintf('You are simulating flow around a field of cylinders\n');
    fprintf('Flow proceeds left to right through the channel containing 2D cylinders\n');
    fprintf('You should see flow wrapping around the cylinders\n');
    fprintf('Try changing the tau (viscosity) to observe differing dynamics\n');
    fprintf('Also try adding cylinders or changing their place in the "give_Me_Problem_Geometry" function\n\n\n');
    
elseif strcmp(choice,'porous1')
    
    fprintf('You are simulating porous media flow\n');
    fprintf('Flow proceeds left to right through the channel containing obstacles\n');
    fprintf('Try changing the porosity (percentPorosity) to observe differing dynamics\n');
    fprintf('NOTE: each simulation creates a RANDOM porous geometry\n\n\n');
    
elseif strcmp(choice,'porous2')
    
    fprintf('You are simulating flow through various porous layers\n');
    fprintf('Flow proceeds left to right through the channel containing obstacles\n');
    fprintf('Try changing the porosity (percentPorosity) to observe differing dynamics\n');
    fprintf('NOTE: each simulation creates a RANDOM porous geometry\n\n\n');

end
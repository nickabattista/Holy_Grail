function lets_do_LBM()

% 2D LATTICE BOLTZMANN (LBM) SIMULATION 
% Author: Nicholas A. Battista
% Created: 11/4/2014
% Modified: 12/2/2014

%  D2Q9 Model:
%  c6  c2   c5  
%    \  |  /    
%  c3- c9 - c1  
%    /  |  \   
%  c7  c4   c8     

%f_i: the probability for site vec(x) to have a particle heading in
%     direction i, at time t. These called discretized probability
%     distribution functions and represent the central link to LBMs.

%LBM Idea: 
%         1. At each timestep the particle densities propogate in each direction (1-8). 
%         2. An equivalent "equilibrium' density is found
%         3. Densities relax towards that state, in proportion governed by
%            tau (parameter related to viscosity).


%Prints key ideas to screen
print_LBM_Info(); 


%
% Simulation Parameters
%
tau=1.4;                      %tau: relaxation parameter related to viscosity
density=0.01;                 %density to be used for initializing whole grid to value 1.0
w1=4/9; w2=1/9; w3=1/36;      %weights for finding equilibrium distribution
nx=256; ny=256;               %number of grid cells
Lx = 1; Ly = 1;               %Size of computational domain
f=repmat(density/9,[nx ny 9]);%Copies density/9 into 9-matrices of size [nx,ny] -> ALLOCATION for all "DIRECTIONS"
f_EQ = f;                     %Initializes F-equilibrium Storage space
grid_size=nx*ny;              %Total number of grid cells
CI= 0:grid_size:7*grid_size;  %Indices to point to FIRST entry of the desired "z-stack" distribution grid      


%
% Chooses which problem to simulate
%
% Possible Choices: 'cylinder1', 'cylinder2', 'channel', 'porous1', 'porous2'
%
choice = 'porous2';
percentPorosity = 0.25;  % Percent of Domain that's Porous (does not matter if not studying porous problem)
[BOUND,deltaU,endTime] = give_Me_Problem_Geometry(choice,nx,ny,percentPorosity); %BOUND: gives geometry, deltaU: gives incremental increase to inlet velocity
print_simulation_info(choice);


%Find Indices of NONZERO Elements, i.e., where "boundary points" IS
ON=find(BOUND); %matrix offset of each Occupied Node


%Offsets Indices for the Different Directions [i.e., levels of F_i=F(:,:,i) ] for known BOUNDARY pts.
TO_REFLECT=[ON+CI(1) ON+CI(2) ON+CI(3) ON+CI(4) ON+CI(5) ON+CI(6) ON+CI(7) ON+CI(8)];
REFLECTED= [ON+CI(3) ON+CI(4) ON+CI(1) ON+CI(2) ON+CI(7) ON+CI(8) ON+CI(5) ON+CI(6)];


%Initialization Parameters
avgU=1;                           %initialize avg. velocity to 1.0
prevAvgU=1;                       %initialize previous-avg. velocity to 1.0
ts=0;                             %initialize starting time to 0
numactivenodes=sum(sum(1-BOUND)); %Finds number of nodes that ARE NOT boundary pts.


% SAVING DATA TO VTK %
print_dump = floor(endTime/50);
ctsave = 0;
% CREATE VIZ_IB2D FOLDER and VISIT FILES
mkdir('vtk_data');
UX = zeros(nx,ny); UY = UX; vorticity = UX(1:end-1,1:end-1); 
print_vtk_files(ctsave,UX,UY,vorticity,Lx,Ly,nx,ny);



%Begin time-stepping!
while ts < endTime
    
    % STREAMING STEP (progate in respective directions)
    f = please_Stream_Distribution(f,nx,ny);
    
    %Densities bouncing back at next timestep
    BOUNCEDBACK=f(TO_REFLECT); 
    
    %vec(rho) = SUM_i f_i -> SUMS EACH DISTRIBUTION MATRIX TOGETHER
    DENSITY=sum(f,3);  %Note: '3' denotes sum over third dimension
    
    %vec(u) = 1/vec(rho) SUM_i (f_i)(e_i) -> CREATES VELOCITY MATRICES
    UX=( sum(f(:,:,[1 5 8]),3)-sum(f(:,:,[3 6 7]),3) ) ./ DENSITY; 
    UY=( sum(f(:,:,[2 5 6]),3)-sum(f(:,:,[4 7 8]),3) ) ./ DENSITY;
    
    %Increase inlet velocity with each time step along left wall
    UX(1,1:ny)=UX(1,1:ny) + deltaU; 
    
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
    f(REFLECTED)=BOUNCEDBACK;
    
    %Updates simulation parameters
    ts=ts+1;   %update time step
    
    % Save files info!
    ctsave = ctsave + 1;
    if mod(ctsave,print_dump) == 0

        % compute vorticity
        dUx_y = UX(1:nx-1,2:ny)-UX(1:nx-1,1:ny-1);
        dUy_x = UY(2:nx,1:ny-1)-UY(1:nx-1,1:ny-1);
        vorticity(1:nx-1,1:ny-1)=( dUy_x - dUx_y )/(2);
        
        % print to vtk
        print_vtk_files(ctsave,UX,UY,vorticity,Lx,Ly,nx,ny);
        fprintf('Simulation Time: %d\n',ts);
    
    end
    
end

%Plots the "steady-state" velocities
plot_Steady_State(UX,UY,BOUND,nx,ny,ts,choice);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to choose what geometry to consider for the simulation
% Returns: Geometry / Increase to inlet velocity for each time-step / endTime
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [BOUND deltaU endTime] = give_Me_Problem_Geometry(choice,nx,ny,percentPorosity)


if strcmp(choice,'cylinder1')

    %CHANNEL FLOW W/ CYLINDER
    a=repmat(-(nx-1)/2:(nx-1)/2,[ny,1]); 
    r = floor(nx/5);
    aR = ceil(nx/5);
    BOUND=( a.^2+(a+aR)'.^2)<r;          %PUTS "1's" within region of Cylinder
    BOUND(1:nx,[1 ny])=1;               %Puts "1's" on Left/Right Boundaries
    deltaU = 0.01;                      %Incremental increase to inlet velocity
    endTime = 2500;
    
elseif strcmp(choice,'cylinder2')

    %CHANNEL FLOW W/ CYLINDER
    a=repmat(-(nx-1)/2:(nx-1)/2,[ny,1]);
    r = floor(nx/5);
    aL = floor(nx/5);
    aR = ceil(nx/5);
    aM = 1;
    B1=  ( ( (a).^2+(a+aR)'.^2)<r );          %PUTS "1's" within region of Cylinder1
    B2=  ( ( (a+aL).^2+(a-aM)'.^2)<r );        %PUTS "1's" within region of Cylinder2
    B3=  ( ( (a-aL).^2+(a-aM)'.^2)<r );        %PUTS "1's" within region of Cylinder1
    BOUND= double(B1)+double(B2)+double(B3); %PUTS together all cylinder geometry
    BOUND(1:nx,[1 ny])=1;                    %Puts "1's" on Left/Right Boundaries
    deltaU = 0.01;                           %Incremental increase to inlet velocity
    endTime = 5000;

elseif strcmp(choice,'channel')
    
    %CHANNEL GEOMETRY
    BOUND=zeros(nx,ny);
    BOUND(1:nx,[1 ny])=1;               %PUTS "1's" on LEFT/RIGHT Boundaries
    deltaU = 0.01;                      %Incremental increase to inlet velocity
    endTime = 2500;


elseif strcmp(choice,'porous1')
    
    %POROUS RANDOM DOMAIN
    BOUND=rand(nx,ny)<percentPorosity;   %PUTS "1's" inside domain randomly if RAND value above percent  
    aS = ceil(nx/5);
    aE = ceil(4*5/nx);
    BOUND(1:aS,:) = 0; 
    BOUND(aE:end,:)=0;
    BOUND(1:nx,[1 ny])=1;                %PUTS "1's" on LEFT/RIGHT Boundaries
    deltaU = 1e-7;                       %Incremental increase to inlet velocity
    endTime = 5000;
    
elseif strcmp(choice,'porous2')
    
    %POROUS RANDOM DOMAIN
    BOUND=rand(nx,ny)<percentPorosity;  %PUTS "1's" inside domain randomly if RAND value above percent              
    BOUND(1:floor(9*nx/31),:) = 0;                   %PUTS "0's" to make open channels through porous structure
    BOUND(floor(7*nx/31):floor(9*nx/31),:) = 0;                   %PUTS "0's" to make open channels through porous structure
    BOUND(floor(13*nx/31):floor(15*nx/31),:) = 0;                 %PUTS "0's" to make open channels through porous structure
    BOUND(floor(19/31*nx):floor(21/31*nx),:) = 0;                 %PUTS "0's" to make open channels through porous structure
    BOUND(floor(25/31*nx):floor(27/31*nx),:)=0;                   %PUTS "0's" to make open channels through porous structure
    BOUND(floor(30/31*nx):end,:) = 0;                %PUTS "0's" to make open channels through porous structure
    BOUND(1:nx,[1 ny])=1;               %PUTS "1's" on LEFT/RIGHT Boundaries
    deltaU = 1e-7;                      %Incremental increase to inlet velocity
    endTime = 5000;

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to stream the distribution function, f.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = please_Stream_Distribution(f,nx,ny)
    
f(:,:,1)=f([nx 1:nx-1],:,1);          %Stream Right

f(:,:,2)=f(:,[ny 1:ny-1],2);          %Stream Up

f(:,:,3)=f([2:nx 1],:,3);             %Stream Left

f(:,:,4)=f(:,[2:ny 1],4);             %Stream Down

f(:,:,5)=f([nx 1:nx-1],[ny 1:ny-1],5);%Stream Right-Up

f(:,:,6)=f([2:nx 1],[ny 1:ny-1],6);   %Stream Left-Up

f(:,:,7)=f([2:nx 1],[2:ny 1],7);      %Stream Left-Down    

f(:,:,8)=f([nx 1:nx-1],[2:ny 1],8);   %Stream Right-Down

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
% Function to PLOT the steady-state velocities, UX and UY, at time-step, ts.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_Steady_State(UX,UY,BOUND,nx,ny,ts,choice)


%Set up grids
xGrid = 1:1:nx;
yGrid = 1:1:ny;

%Scale Factor for Velocity Vectors
scale = 5;

%Streamlines
yStart = floor(nx/3):2:floor(2*nx/3);
xStart = 2*ones(size(yStart));
options = [0.01 10000];

%Compute magnitude of velocity
velMag = sqrt( UX.*UX + UY.*UY );

%Compute vorticity (approximate vorticity = dUy/dx - dYx/dy )
vorticity = (UY - UX)/2;
dUx_y = UX(1:nx-1,2:ny)-UX(1:nx-1,1:ny-1);
dUy_x = UY(2:nx,1:ny-1)-UY(1:nx-1,1:ny-1);
vorticity(1:nx-1,1:ny-1)=( dUy_x - dUx_y )/(2); 



if strcmp(choice,'channel')
    
    figure(1);
    subplot(1,2,1)
    colormap(gray(2));
    image(2-BOUND');hold on;
    quiver(xGrid,yGrid,UX(xGrid,:)',UY(yGrid,:)');
    title(['Velocity Field at',num2str(ts),'\Deltat']);
    xlabel('x');
    ylabel('y');
    
    xPts = [floor(nx/4) floor(2*nx/4) floor(3*nx/4)];
    yVec = 1:ny;
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
    axis([0 nx 0 1.05*max(max(mat))]);
    
elseif ( strcmp(choice,'cylinder1') || strcmp(choice,'cylinder2') )
    
    figure(1);
    subplot(1,2,1)
    colormap(gray(2));
    image(2-BOUND');hold on;
    quiver(xGrid,yGrid,UX(xGrid,:)',UY(yGrid,:)');
    title(['Velocity Field at',num2str(ts),'\Deltat']);
    xlabel('x');
    ylabel('y');
    
    xPts = [floor(nx/5) floor(2*nx/5) floor(3*nx/5) floor(4*nx/5)];
    yVec = 1:ny;
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
    axis([0 nx 0 1.05*max(max(mat))]);
       
elseif strcmp(choice,'porous1')
    
    figure(1);
    subplot(1,2,1);
    colormap(gray(2));
    uMag = sqrt( UX.*UX + UY.*UY );
    scaley = 5.5/max( max( uMag ));
    image(2-BOUND');hold on;
    quiver(xGrid(2:end),yGrid(1:end),scaley*UX(xGrid(2:end),:)',scaley*UY(yGrid(2:end),:)','AutoScale','off');
    title(['Velocity Field at',num2str(ts),'\Deltat']);
    xlabel('x');
    ylabel('y');
    axis([1 nx 1 ny]);
    
    xPts = [floor(nx/15) floor(nx/5) nx-floor(nx/5) nx-floor(nx/15)];
    %xPts = [2 6 26 31];
    yVec = 1:ny;
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
    axis([0 nx 0 1.05*max(max(mat))]);
    
elseif strcmp(choice,'porous2')
    
    figure(1);
    subplot(1,2,1);
    colormap(gray(2));
    uMag = sqrt( UX.*UX + UY.*UY );
    scaley = 5.5/max( max( uMag ));
    image(2-BOUND');hold on;
    quiver(xGrid(2:end),yGrid(1:end),scaley*UX(xGrid(2:end),:)',scaley*UY(yGrid(2:end),:)','AutoScale','off');
    title(['Velocity Field at ',num2str(ts),'\Deltat']);
    xlabel('x');
    ylabel('y');
    axis([1 nx 1 ny]);
    
    xPts = [floor(nx/5) floor(2*nx/5) floor(3*nx/5) floor(4*nx/5)];
    %xPts = [8 14 20 26];
    yVec = 1:ny;
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
    axis([0 nx 0 1.05*max(max(mat))]);
   
end

figure(2)
subplot(1,2,1)
    hold off
    contourf(xGrid,yGrid,velMag',7);hold on;
    quiver(xGrid,yGrid,scale*UX(1:nx,:)',scale*UY(1:nx,:)','k','AutoScale','off');hold on;
    title('Velocity Magnitude Colormap');
    axis equal
    axis([1 nx 1 ny]);

subplot(1,2,2)
    hold off
    contourf(xGrid,yGrid,vorticity',3); hold on;
    h = streamline(xGrid,yGrid,scale*UX(1:nx,:)',scale*UY(1:nx,:)',xStart,yStart,options); hold on;
    set(h,'linewidth',1.25,'color','k');
    plot(xStart,yStart,'r.','MarkerSize',6); hold on;
    title('Vorticity Colormap and Streamlines');
    axis equal
    axis([1 nx 1 ny]);
    



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to print Lattice Boltzmann key ideas to screen.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_LBM_Info()

fprintf('\n\n 2D LATTICE BOLTZMANN (LBM) SIMULATION \n');
fprintf('Author: Nicholas A. Battista\n');
fprintf('Created: 11/4/2014\n');
fprintf('Modified: 12/2/2014\n\n');
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
    fprintf('Flow proceeds left to right through the channel containing a 2D cylinder\n');
    fprintf('You should see flow wrapping around the cylinders\n');
    fprintf('Try changing the tau (viscosity) to observe differing dynamics\n');
    fprintf('Also try adding cylinders or changing their place in the "give_Me_Problem_Geometry" function\n\n\n');
    
elseif strcmp(choice,'porous1');
    
    fprintf('You are simulating porous media flow\n');
    fprintf('Flow proceeds left to right through the channel containing obstacles\n');
    fprintf('Try changing the porosity (percentPorosity) to observe differing dynamics\n');
    fprintf('NOTE: each simulation creates a RANDOM porous geometry\n\n\n');
    
elseif strcmp(choice,'porous2');
    
    fprintf('You are simulating flow through various porous layers\n');
    fprintf('Flow proceeds left to right through the channel containing obstacles\n');
    fprintf('Try changing the porosity (percentPorosity) to observe differing dynamics\n');
    fprintf('NOTE: each simulation creates a RANDOM porous geometry\n\n\n');

end
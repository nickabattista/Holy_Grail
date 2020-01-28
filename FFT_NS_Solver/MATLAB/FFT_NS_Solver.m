function FFT_NS_Solver()

%
% Solves the Navier-Stokes equations in the Vorticity-Stream Function
% formulation using a pseudo-spectral approach w/ FFT
%
% Author: Nicholas A. Battista
% Created: Novermber 29, 2014
% Modified: September 28, 2019
% 
% Equations of Motion:
% D (Vorticity) /Dt = nu*Laplacian(Vorticity)  
% Laplacian(Psi) = - Vorticity                                                       
%
%      Real Space Variables                   Fourier (Frequency) Space                                                          
%       SteamFunction: Psi                     StreamFunction: Psi_hat
% Velocity: u = Psi_y & v = -Psi_x              Velocity: u_hat ,v_hat
%         Vorticity: Vort                        Vorticity: Vort_hat
%
%
% IDEA: for each time-step
%       1. Solve Poisson problem for Psi (in Fourier space)
%       2. Compute nonlinear advection term by finding u and v in real
%          variables by doing an inverse-FFT, compute advection, transform
%          back to Fourier space
%       3. Time-step vort_hat in frequency space using a semi-implicit
%          Crank-Nicholson scheme (explicit for nonlinear adv. term, implicit
%          for viscous term)
%

% Print key fluid solver ideas to screen
print_FFT_NS_Info();

%
% Simulation Parameters
%
nu=1.0e-3;  % kinematic viscosity (dynamic viscosity/density)
NX = 256;   % # of grid points in x 
NY = 256;   % # of grid points in y   
LX = 1;     % 'Length' of x-Domain
LY = 1;     % 'Length' of y-Domain      

%
% Choose initial vorticity state
% Choices:  'half', 'qtrs', 'rand', 'bubble3', 'bubbleSplit','bubble1'
%
choice='bubble3';
[vort_hat,dt,tFinal,plot_dump] = please_Give_Initial_Vorticity_State(choice,NX,NY);

%
% Initialize wavenumber storage for fourier exponentials
%
[kMatx, kMaty, kLaplace] = please_Give_Wavenumber_Matrices(NX,NY);


t=0.0;            %Initialize time to 0.0
fprintf('Simulation Time: %d\n',t);
nTot = tFinal/dt; %Total number of time-steps
for n=0:nTot      %Enter Time-Stepping Loop!
    
    % Printing zero-th time-step
    if n==0
        
        %Solve Poisson Equation for Stream Function, psi
        psi_hat = please_Solve_Poission(vort_hat,kMatx,kMaty,NX,NY);

        %Find Velocity components via derivatives on the stream function, psi
        u  =real(ifft2( kMaty.*psi_hat));        % Compute  y derivative of stream function ==> u = psi_y
        v  =real(ifft2(-kMatx.*psi_hat));        % Compute -x derivative of stream function ==> v = -psi_x
        
        % SAVING DATA TO VTK %
        ctsave = 0; % total # of time-steps
        pSave = 0;  % time-step data counter
        
        % CREATE VIZ_IB2D FOLDER and VISIT FILES
        mkdir('vtk_data');
            
        % Transform back to real space via Inverse-FFT
        vort_real=real(ifft2(vort_hat));

        % Save .vtk data!
        % Note: switch order of u and v in this function bc of notation-> f(x,y) here rather than matrix convention of y(row,col) w/ y=row, x=col
        print_vtk_files(pSave,v,u,vort_real,LX,LY,NX,NY);
   
    else
    
        %Solve Poisson Equation for Stream Function, psi
        psi_hat = please_Solve_Poission(vort_hat,kMatx,kMaty,NX,NY);

        %Find Velocity components via derivatives on the stream function, psi
        u  =real(ifft2( kMaty.*psi_hat));        % Compute  y derivative of stream function ==> u = psi_y
        v  =real(ifft2(-kMatx.*psi_hat));        % Compute -x derivative of stream function ==> v = -psi_x

        %Compute derivatives of voriticty to be "advection operated" on
        vort_X=real(ifft2( kMatx.*vort_hat  ));  % Compute  x derivative of vorticity
        vort_Y=real(ifft2( kMaty.*vort_hat  ));  % Compute  y derivative of vorticity

        %Compute nonlinear part of advection term
        advect = u.*vort_X + v.*vort_Y;          % Advection Operator on Vorticity: (u,v).grad(vorticity)   
        advect_hat = fft2(advect);               % Transform advection (nonlinear) term of material derivative to frequency space

        % Compute Solution at the next step (uses Crank-Nicholson Time-Stepping)
        vort_hat = please_Perform_Crank_Nicholson_Semi_Implict(dt,nu,NX,NY,kLaplace,advect_hat,vort_hat);
        %vort_hat = ((1/dt + 0.5*nu*kLaplace)./(1/dt - 0.5*nu*kLaplace)).*vort_hat - (1./(1/dt - 0.5*nu*kLaplace)).*advect_hat;

        % Update time
        t=t+dt; 

        % UNCOMMENT BELOW to plot the vorticity colormap directly in MATLAB
%         if mod(n,plot_dump) == 0
%             
%             % Transform back to real space via Inverse-FFT
%             vort_real=real(ifft2(vort_hat));
%             
%             % Compute smaller matrices for velocity vector field plots
%             newSize = 200;       %new desired size of vector field to plot (i.e., instead of 128x128, newSize x newSize for visual appeal)
%             [u,v,xVals,yVals] = please_Give_Me_Smaller_Velocity_Field_Mats(u,v,NX,NY,newSize);
%             
%             contourf(vort_real,10); hold on;
%             quiver(xVals(1:end),yVals(1:end),u,v); hold on;
%             
%             colormap('jet'); colorbar; 
%             title(['Vorticity and Velocity Field at time ',num2str(t)]);
%             axis([1 NX 1 NY]);
%             drawnow;
%             %pause(0.01);
%             
%         end

        % Save files info to .vtk format!
        ctsave = ctsave + 1;
        if mod(ctsave,plot_dump) == 0

            % increment Print counter
            pSave = pSave + 1;
            
            % Transform back to real space via Inverse-FFT
            vort_real=real(ifft2(vort_hat));

            % Save .vtk data!
            % Note: switch order of u and v in this function bc of notation-> f(x,y) here rather than matrix convention of y(row,col) w/ y=row, x=col
            print_vtk_files(pSave,v,u,vort_real,LX,LY,NX,NY);

            % Plot simulation time
            fprintf('Simulation Time: %d\n',t);

        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to choose initial vorticity state
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [vort_hat,dt,tFinal,plot_dump] = please_Give_Initial_Vorticity_State(choice,NX,NY)

if strcmp(choice,'half')
    
    %
    % USE RECTANGLE: Lx = 2Ly, Nx = 2Ny
    %
    
    % radii of vortex regions (given in terms of mesh widths)
    radius1 = 0.3*NY;
    radius2 = 0.15*NY;
    
    % stack vectors to create grids of indices
    a1=repmat(-(NX-1)/2:(NX-1)/2,[NY,1]); 
    a2=repmat(-(NY-1)/2:(NY-1)/2,[NX,1]);    
    
    % Amount to translate region from middle of domain
    aR = floor(0.25*NX);    

    % Form circular regions of vorticity
    b1 = ( (a1+aR).^2+(a2)'.^2) < radius1^2; 
    b2 = ( (a1-aR).^2+(a2)'.^2) < radius2^2; 

    % Convert boolean matrix to matrix of double values
    % Note assuming no overlapping vorticity regions here
    b1 = double(b1) + double(b2);
    
    % Find values where vorticity is
    [r1,c1]=find(b1==1);
    
    vort = zeros(NX,NY);
    for i=1:length(r1)
        if c1(i) > NX/2
            vort(c1(i),r1(i))= -0.05;%0.5*(rand(1)+1);
        else
            vort(c1(i),r1(i))= -0.1;%-0.5*(rand(1)+1);
        end
    end  
    
    dt=1e-2;        % time step
    tFinal = 5;     % final time
    plot_dump=20;   % interval for plots
    

elseif strcmp(choice,'qtrs')
    
    %
    % SQUARE: Lx = Ly, Nx = Ny
    %
    
    % radii of vortex regions (given in terms of mesh widths)
    radius11 = 0.2*NY;
    radius12 = 0.2*NY;
    radius21 = 0.2*NY;
    radius22 = 0.2*NY;    
    
    % stack vectors to create grids of indices
    a1=repmat(-(NX-1)/2:(NX-1)/2,[NY,1]); 
    a2=repmat(-(NY-1)/2:(NY-1)/2,[NX,1]);    
    
    % Amount to translate region from middle of domain
    aR = floor(0.25*NX); 
    aU = floor(0.25*NY);

    % Form circular regions of vorticity
    b1 = ( (a1+aR).^2+(a2+aU)'.^2) < radius11^2; 
    b2 = ( (a1-aR).^2+(a2-aU)'.^2) < radius12^2; 
    b3 = ( (a1+aR).^2+(a2-aU)'.^2) < radius21^2; 
    b4 = ( (a1-aR).^2+(a2+aU)'.^2) < radius22^2; 
    
    % Convert boolean matrix to matrix of double values
    b1 = double(b1) + double(b2) + double(b3) + double(b4);
    
    % Find values where vorticity is
    [r1,c1]=find(b1==1);
    
    vort = zeros(NX,NY);
    for i=1:length(r1)
        if c1(i) > NX/2
            if r1(i) > NY/2
                vort(c1(i),r1(i))= 0.1;%0.5*(rand(1)+1);
            else
                vort(c1(i),r1(i))= -0.1;%0.5*(rand(1)+1);
            end
        else
            if r1(i) > NY/2
                vort(c1(i),r1(i))= 0.1;%0.5*(rand(1)+1);
            else
                vort(c1(i),r1(i))= -0.1;%0.5*(rand(1)+1);
            end
        end
    end  
    
    dt=1e-2;        % time step
    tFinal = 5;     % final time
    plot_dump=20;   % interval for plots
    
   
elseif strcmp(choice,'rand')
    
    %
    % Any domain is fine.
    %
    
    vort = 2*rand(NX,NY)-1;
    dt=1e-1;       % time step
    tFinal = 1000; % final time
    plot_dump=25;  % interval for plots

elseif strcmp(choice,'bubble1')
    
    %
    % SQUARE: Lx = Ly, Nx = Ny
    %
    
    % radius of bubble (centered in middle of domain, given in terms of mesh widths)
    radius1 = 0.25*NX;
    
    % stack vectors to create grids of indices
    a1=repmat(-(NX-1)/2:(NX-1)/2,[NY,1]); 
    a2=repmat(-(NY-1)/2:(NY-1)/2,[NX,1]);    

    % Form circular regions of vorticity
    b1 = ( (a1).^2+(a2)'.^2) < radius1^2;         % region at center of domain
    
    % Initialize vorticity in grid to random values between -1,1
    vort = 2*rand(NX,NY)-1;

    % Find values where largest region is
    [r1,c1]=find(b1==1);
    for i=1:length(r1)
        vort(c1(i),r1(i))=  0.6;
    end
    
    dt = 1e-2;      % time step
    tFinal = 30;    % final time
    plot_dump= 50;  % interval for plots
    
elseif strcmp(choice,'bubbleSplit')

    %
    % SQUARE: Lx = Ly, Nx = Ny
    %
    
    % radius of bubble (centered in middle of domain, given in terms of mesh widths)
    radius1 = 0.25*NX;
    
    % stack vectors to create grids of indices
    a1=repmat(-(NX-1)/2:(NX-1)/2,[NY,1]); 
    a2=repmat(-(NY-1)/2:(NY-1)/2,[NX,1]);    

    % Form circular regions of vorticity
    b1 = ( (a1).^2+(a2)'.^2) < radius1^2;         % region at center of domain
    
    % Initialize vorticity in grid to random values between -1,1
    vort = 2*rand(NX,NY)-1;

    % Find values where largest region is
    [r1,c1]=find(b1==1);
    for i=1:length(r1)
        if c1(i) < NX/2
            vort(c1(i),r1(i))=  -0.5*rand(1);
        else
            vort(c1(i),r1(i))=  0.5*rand(1);
        end
    end
    
    dt = 1e-2;      % time step
    tFinal = 30;    % final time
    plot_dump= 50;  % interval for plots
    
    
elseif strcmp(choice,'bubble3')
    
    %
    % SQUARE: Lx = Ly, Nx = Ny
    %
    
    % radius of bubble (centered in middle of domain, given in terms of mesh widths)
    radius1 = 0.25*NX;
    radius2 = 0.175*NX;
    radius3 = 0.10*NX;
    
    % stack vectors to create grids of indices
    a1=repmat(-(NX-1)/2:(NX-1)/2,[NY,1]); 
    a2=repmat(-(NY-1)/2:(NY-1)/2,[NX,1]);    
    
    % Amount to translate region from middle of domain
    aD = floor(0.10*NX); 
    aR = floor(0.15*NX); 

    % Form circular regions of vorticity
    b1 = ( (a1).^2+(a2)'.^2) < radius1^2;         % region at center of domain
    b2 = ( (a1).^2+(a2-aD)'.^2) < radius2^2;      % shift 2nd region down from center
    b3 = ( (a1+aR).^2+(a2-2*aD)'.^2) < radius3^2; % shift 3rd region down/right from center
    
    % Initialize vorticity in grid to random values between -1,1
    vort = 2*rand(NX,NY)-1;

    % Find values where largest region is
    [r1,c1]=find(b1==1);
    for i=1:length(r1)
        vort(c1(i),r1(i))=  0.4;
    end
    
    % Find values where 2ND largest region is
    [r1,c1]=find(b2==1);
    for i=1:length(r1)
        vort(c1(i),r1(i))=  -0.5;
    end
    
    % Find values where 3RD largest region is (smallest)
    [r1,c1]=find(b3==1);
    for i=1:length(r1)
        vort(c1(i),r1(i))=  0.5;
    end
    
    dt = 1e-2;      % time step
    tFinal = 30;    % final time
    plot_dump= 50;  % interval for plots
    
end

% Finally transform initial vorticity state to frequency space using FFT
vort_hat=fft2(vort);  

% Print simulation information
print_Simulation_Info(choice);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: Initializes Wavenumber Matrices for FFT
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [kMatx, kMaty, kLaplace] = please_Give_Wavenumber_Matrices(NX,NY)

kMatx = zeros(NX,NY);
kMaty = kMatx;

rowVec = [0:NY/2 (-NY/2+1):1:-1];
colVec = [0:NX/2 (-NX/2+1):1:-1]';

%Makes wavenumber matrix in x
for i=1:NX
   kMatx(i,:) = 1i*rowVec;
end

%Makes wavenumber matrix in y (NOTE: if Nx=Ny, kMatx = kMaty')
for j=1:NY
   kMaty(:,j) = 1i*colVec; 
end

% Laplacian in Fourier space
kLaplace=kMatx.^2+kMaty.^2;        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to solve poisson problem, Laplacian(psi) = -Vorticity
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function psi_hat = please_Solve_Poission(w_hat,kx,ky,NX,NY)

psi_hat = zeros(NX,NY); %Initialize solution matrix
kVecX = kx(1,:);        %Gives row vector from kx
kVecY = ky(:,1);        %Gives column vector from ky

for i = 1:NX
    for j = 1:NY
        if ( i+j > 2 )
            psi_hat(i,j) = -w_hat(i,j)/( ( kVecX(j)^2+ kVecY(i)^2 ) ); % "inversion step"
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to take every few entries from velocity field matrices for
% plotting the field. (For aesthetic purposes)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [u,v,xVals,yVals] = please_Give_Me_Smaller_Velocity_Field_Mats(uOld,vOld,NX,NY,newSize)

iterX = NX/newSize;
xVals = 1:iterX:NX;

iterY = NY/newSize;
yVals = 1:iterY:NY;

if ( mod(NX,newSize) > 0 ) || ( mod(NY,newSize) > 0 )
    
    %If numbers don't divide properly throw this flag and use original
    fprintf('Will not be able to reSize velocity field matrices.');
    xVals = 1:NX;
    yVals = 1:NY;
    u = uOld;
    v = vOld;
    
else
    
    u = zeros(length(xVals),length(yVals));  %initialize new u
    v = u;                                   %initialize new v
    n = 0; m = 0;                            %initializing new counter for resizing
    for i=1:NX
        if mod(i,iterX)==1
            n = n+1;
            for j=1:NY
                if ( mod(j,iterY) == 1 )
                    m = m+1;
                    u(n,m) = uOld(i,j);
                    v(n,m) = vOld(i,j);
                end
            end
            m=0;
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to perform one time-step of Crank-Nicholson Semi-Implicit
% timestepping routine to get next time-step's vorticity coefficients in
% fourier (frequency space). 
%
% Note: 1. The nonlinear advection is handled explicitly
%       2. The viscous term is handled implictly
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vort_hat = please_Perform_Crank_Nicholson_Semi_Implict(dt,nu,NX,NY,kLaplace,advect_hat,vort_hat)

    for i=1:NX
        for j=1:NY

            %Crank-Nicholson Semi-Implicit Time-step
            vort_hat(i,j) = ( (1 + dt/2*nu*kLaplace(i,j) )*vort_hat(i,j) - dt*advect_hat(i,j) ) / (  1 - dt/2*nu*kLaplace(i,j) );

        end
    end
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to print information about fluid solver
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_FFT_NS_Info()

fprintf('\n_________________________________________________________________________\n\n');
fprintf(' \nSolves the Navier-Stokes equations in the Vorticity-Stream Function \n');
fprintf(' formulation using a pseudo-spectral approach w/ FFT \n\n');
fprintf(' Author: Nicholas A. Battista \n');
fprintf(' Created: Novermber 29, 2014 \n');
fprintf(' Modified: December 5, 2014 \n\n');
fprintf(' Equations of Motion: \n');
fprintf(' D (Vorticity) /Dt = nu*Laplacian(Vorticity)  \n');
fprintf(' Laplacian(Psi) = - Vorticity                 \n\n');                                     
fprintf('      Real Space Variables                   Fourier (Frequency) Space              \n');                                            
fprintf('       SteamFunction: Psi                     StreamFunction: Psi_hat \n');
fprintf(' Velocity: u = Psi_y & v = -Psi_x              Velocity: u_hat ,v_hat \n');
fprintf('         Vorticity: Vort                        Vorticity: Vort_hat \n\n');
fprintf('_________________________________________________________________________\n\n');
fprintf(' IDEA: for each time-step \n');
fprintf('       1. Solve Poisson problem for Psi (in Fourier space)\n');
fprintf('       2. Compute nonlinear advection term by finding u and v in real \n');
fprintf('          variables by doing an inverse-FFT, compute advection, transform \n');
fprintf('          back to Fourier space \n');
fprintf('       3. Time-step vort_hat in frequency space using a semi-implicit \n');
fprintf('          Crank-Nicholson scheme (explicit for nonlinear adv. term, implicit \n');
fprintf('          for viscous term) \n');
fprintf('_________________________________________________________________________\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to print information about specific simulation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_Simulation_Info(choice)

if strcmp(choice,'bubble1')
   
    fprintf('You are simulating one dense region of CW vorticity in a bed of random vorticity values\n');
    fprintf('Try changing the kinematic viscosity to see how flow changes\n');
    fprintf('_________________________________________________________________________\n\n');

    
elseif strcmp(choice,'bubble3')
    
    fprintf('You are simulating three nested regions of Vorticity (CW,CCW,CW) in a bed of random vorticity values\n');
    fprintf('Try changing the position of the nested vortices in the "please_Give_Initial_State" function\n');
    fprintf('Try changing the kinematic viscosity to see how flow changes\n');
    fprintf('_________________________________________________________________________\n\n');


elseif strcmp(choice,'bubbleSplit')
    
    fprintf('You are simulating two vortices which are very close\n');
    fprintf('Try changing the initial vorticity distribution on the left or right side\n');
    fprintf('Try changing the kinematic viscosity to see how the flow changes\n');
    fprintf('_________________________________________________________________________\n\n');

    
elseif strcmp(choice,'qtrs')
    
    fprintf('You are simulating 4 squares of differing vorticity\n');
    fprintf('Try changing the initial vorticity in each square to see how the dynamics change\n');
    fprintf('Try changing the kinematic viscosity to see how the flow changes\n');
    fprintf('_________________________________________________________________________\n\n');

    
elseif strcmp(choice,'half')
    
    fprintf('You are simulating two half planes w/ opposite sign vorticity\n');
    fprintf('Try changing the kinematic viscosity to see how the flow changes\n');
    fprintf('_________________________________________________________________________\n\n');


elseif strcmp(choice,'rand')
   
    fprintf('You are simulating a field of random vorticity values\n');
    fprintf('Try changing the kinematic viscosity to see how the flow changes\n');
    fprintf('_________________________________________________________________________\n\n');

     
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: from a u.XXXX.vtk file, we will extract U,V (x-Velocity and
% y-Velocity components), compute the background vorticity, and then return
% it to initialize a simulation.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function initial_vorticity = give_Initial_Vorticity_From_Velocity_Field_VTK_File()

%
% Grid Parameters (Make sure to match from FFT_NS_Solver.m)
%
Lx = 1;     % x-Length of Computational Grid
Ly = 1;     % y-Length of Computational Grid
Nx = 256;   % Grid resolution in x-Direction
Ny = 256;   % Grid resolution in y-Direction
dx = Lx/Nx; % Spatial step-size in x
dy = Ly/Ny; % Spatial step-size in y

%
% Get back components of velocity field, U,V, e.g., \vec{u} = (U,V)
% 
% Note: U,V are both matrices
%
simNumsString = '0070';
[U,V] = read_Eulerian_Velocity_Field_vtk(simNumsString);

% Note the u.0070.vtk case was performed at 512x512 resolution, hence we
% will downsample for the velocity field, e.g., for 256x256 choose every
% other point
[lenY,~] = size(U);
downsample = lenY/Ny;
U = U(1:downsample:end,1:downsample:end);
V = V(1:downsample:end,1:downsample:end);


%
% Compute Vorticity from Velocity Components
%
initial_vorticity = compute_Vorticity(U,V,dx,dy);

%
% Transpose the vorticity
%
initial_vorticity = initial_vorticity';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: computes the VORTICITY from the vector field's components, U and V
%           e.g., vec{u} = (U,V)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vorticity = compute_Vorticity(U,V,dx,dy)

% Get Size of Matrix
[Ny,Nx] = size(U);

% Initialize
Uy = zeros(Ny,Nx);
Vx = Uy;

%
% Find partial derivative with respect to x of vertical component of
% velocity, V
%
for i=1:Nx
    
    if i==1
        Vx(:,i) = ( V(1:end,2) - V(1:end,end) ) / (2*dx);
    elseif i==Nx
        Vx(:,i) = ( V(1:end,1) - V(1:end,i-1) ) / (2*dx);
    else
        Vx(:,i) = ( V(1:end,i+1) - V(1:end,i-1) ) / (2*dx);
    end
end

%
% Find partial derivative with respect to y of horizontal component of
% velocity, U
%
for j=1:Ny
    if j==1
        Uy(j,:) = ( U(2,1:end) - U(end,1:end) ) / (2*dy);
    elseif j==Ny
        Uy(j,:) = ( U(1,1:end) - U(j-1,1:end) ) / (2*dy);
    else
        Uy(j,:) = ( U(j+1,1:end) - U(j-1,1:end) ) / (2*dy);
    end
end


% Take difference to find vorticity ( Curl(vec{u}) in 2D )
vorticity = Vx - Uy;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: Reads in the velocity data field from .vtk format
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U,V] = read_Eulerian_Velocity_Field_vtk(simNums)

filename = ['u.' num2str(simNums) '.vtk'];  

fileID = fopen(filename);
if ( fileID== -1 )
    error('\nCANNOT OPEN THE FILE!');
end

str = fgets(fileID); %-1 if eof
if ~strcmp( str(3:5),'vtk')
    error('\nNot in proper VTK format');
end

% read in the header info %
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);

% Store grid info
Nx = sscanf(str,'%*s %f %*f %*s',1);
Ny = sscanf(str,'%*s %*f %f %*s',1);


% bypass lines in header %
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);
str = fgets(fileID);


% get formatting for reading in data from .vtk in fscanf %
strVec = '%f';
for i=2:3*Nx
    strVec = [strVec ' %f'];
end

% read in the vertices %
[e_Data,count] = fscanf(fileID,strVec,3*Nx*Ny);
if count ~= 3*Nx*Ny
   error('Problem reading in Eulerian Data.'); 
end

% reshape the matrix into desired data type %
e_Data = reshape(e_Data, 3, count/3); % Reshape (3*Nx*Nx,1) vector to (Nx*Nx,3) matrix
e_Data = e_Data';                     % Store vertices in new matrix

U = e_Data(:,1);       % Store U data
V = e_Data(:,2);       % Store V data

U = reshape(U,Nx,Ny)';  % Reshape (Nx*Nx,1) matrix to  (Nx,Nx)
V = reshape(V,Nx,Ny)';  % Reshape (Nx*Nx,1) matrix to  (Nx,Nx)
 
fclose(fileID);         % Closes the data file.

clear filename fileID str strVec count analysis_path e_Data Nx;


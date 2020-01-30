%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: gives appropriate string number for filename in printing the
% .vtk files.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_vtk_files(ctsave,U,V,vorticity,Lx,Ly,nx,ny)

%Give spacing for grid
dx = Lx/(nx-1); 
dy = Ly/(ny-1);

%Go into vtk_data directory
cd('vtk_data');

%Find string number for storing files
strNUM = give_String_Number_For_VTK(ctsave);

%Prints x-Velocity Component
confName = ['uX.' strNUM '.vtk'];
savevtk_scalar(U, confName, 'uX',dx,dy);

%Prints y-Velocity Component
confName = ['uY.' strNUM '.vtk'];
savevtk_scalar(V, confName, 'uY',dx,dy);

%Prints Mag. of Velocity 
confName = ['uMag.' strNUM '.vtk'];
uMag = sqrt(U.^2 + V.^2);
savevtk_scalar(uMag, confName, 'uMag',dx,dy);

%Prints Vorticity
confName = ['Omega.' strNUM '.vtk'];
savevtk_scalar(vorticity, confName, 'Omega',dx,dy);

%Compute/Prints div( vec{u} ).
%divU = compute_Divergence_of_Vector_Field(U,V,dx,dy);
%confName = ['DivU.' strNUM '.vtk'];
%savevtk_scalar(divU, confName, 'DivU',dx,dy);

%Print VECTOR DATA (i.e., velocity data) to .vtk file
velocityName = ['u.' strNUM '.vtk'];
savevtk_vector(U, V, velocityName, 'u',dx,dy)

%Get out of vtk_data folder
cd ..

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: computes the divergence of vector field vec{u} = (U,V)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function divU = compute_Divergence_of_Vector_Field(U,V,dx,dy)

% Get Size of Matrix
[Ny,Nx] = size(U);

% Initialize
Ux = zeros(Nx-1,Ny-1);
Vy = Ux;

for i=2:Nx-1
    Ux(:,i-1) = ( U(1:end-1,i+1) - U(1:end-1,i-1) ) / (2*dx);
end

for j=2:Ny-1
    Vy(j-1,:) = ( V(j+1,1:end-1) - V(j-1,1:end-1) ) / (2*dy);
end

% Add to find the divergence of the vector field
divU = Ux + Vy;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: prints matrix vector data to vtk formated file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function savevtk_vector(X, Y, filename, vectorName,dx,dy)
%  savevtkvector Save a 3-D vector array in VTK format
%  savevtkvector(X,Y,Z,filename) saves a 3-D vector of any size to
%  filename in VTK format. X, Y and Z should be arrays of the same
%  size, each storing speeds in the a single Cartesian directions.
    if (size(X) ~= size(Y))
        fprint('Error: velocity arrays of unequal size\n'); return;
    end
    [nx, ny, nz] = size(X);
    fid = fopen(filename, 'wt');
    fprintf(fid, '# vtk DataFile Version 2.0\n');
    fprintf(fid, 'Comment goes here\n');
    fprintf(fid, 'ASCII\n');
    fprintf(fid, '\n');
    fprintf(fid, 'DATASET STRUCTURED_POINTS\n');
    fprintf(fid, 'DIMENSIONS    %d   %d   %d\n', nx, ny, nz);
    fprintf(fid, '\n');
    fprintf(fid, 'ORIGIN    0.000   0.000   0.000\n');
    %fprintf(fid, 'SPACING   1.000   1.000   1.000\n'); if want [1,32]x[1,32] rather than [0,Lx]x[0,Ly]
    fprintf(fid, ['SPACING   ' num2str(dx) '  '  num2str(dy) '   1.000\n']);
    fprintf(fid, '\n');
    fprintf(fid, 'POINT_DATA   %d\n', nx*ny);
    fprintf(fid, ['VECTORS ' vectorName ' double\n']);
    fprintf(fid, '\n');
    for a=1:nz
        for b=1:ny
            for c=1:nx
                fprintf(fid, '%f ', X(c,b,1));
                fprintf(fid, '%f ', Y(c,b,1));
                fprintf(fid, '%f ', 0);
            end
            fprintf(fid, '\n');
        end
    end
    fclose(fid);
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: prints scalar matrix to vtk formated file
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function savevtk_scalar(array, filename, colorMap,dx,dy)
%  savevtk Save a 3-D scalar array in VTK format.
%  savevtk(array, filename) saves a 3-D array of any size to
%  filename in VTK format.
    [nx, ny, nz] = size(array);
    fid = fopen(filename, 'wt');
    fprintf(fid, '# vtk DataFile Version 2.0\n');
    fprintf(fid, 'Comment goes here\n');
    fprintf(fid, 'ASCII\n');
    fprintf(fid, '\n');
    fprintf(fid, 'DATASET STRUCTURED_POINTS\n');
    fprintf(fid, 'DIMENSIONS    %d   %d   %d\n', nx, ny, nz);
    fprintf(fid, '\n');
    fprintf(fid, 'ORIGIN    0.000   0.000   0.000\n');
    %fprintf(fid, 'SPACING   1.000   1.000   1.000\n'); if want [1,32]x[1,32] rather than [0,Lx]x[0,Ly]
    fprintf(fid, ['SPACING   ' num2str(dx) '  '  num2str(dy) '   1.000\n']);
    fprintf(fid, '\n');
    fprintf(fid, 'POINT_DATA   %d\n', nx*ny*nz);
    fprintf(fid, ['SCALARS ' colorMap ' double\n']);
    fprintf(fid, 'LOOKUP_TABLE default\n');
    fprintf(fid, '\n');
    for a=1:nz
        for b=1:ny
            for c=1:nx
                fprintf(fid, '%d ', array(c,b,a));
            end
            fprintf(fid, '\n');
        end
    end
    fclose(fid);
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: gives appropriate string number for filename in printing the
% .vtk files.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function strNUM = give_String_Number_For_VTK(num)

%num: # of file to be printed

if num < 10
    strNUM = ['000' num2str(num)];
elseif num < 100
    strNUM = ['00' num2str(num)];
elseif num<1000
    strNUM = ['0' num2str(num)];
else
    strNUM = num2str(num);
end

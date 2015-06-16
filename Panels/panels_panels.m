function panels_panels()

%
%  Panel Method: Here used to solve the incompressible potential flow 
%                equations in 2D.
%
%  Author: Nicholas A. Battista
%  Created: January 6, 2015
%  Modified: February 5, 2015
%
%  Assumptions for Incompressible Potential Flow
%                1. Inviscid
%                2. Incompressible div(V) = 0
%                3. Irrotational   curl(V) = 0
%                4. Steady         partial(u)/partial(t) = 0
%
%  What it does: 
%                This method finds the lift and drag coefficients around an
%                airfoil shape, chosen by the user. It also computes the
%                pressure distribution over the airfoil as well.
%

% Prints simulation info to screen %
please_Print_Sim_Info()


% Create Airfoil Geometry %
w = 1.0; ds = w/25;
[x,Y] = please_Give_Me_Airfoil_Geometry(w,ds);


% OR
% 
% READ IN INPUT DATA!
%[x,Y] = please_Read_In_Data();



%CHOOSE GEOMETRY

scale_Vec = 0.1:0.1:0.5;%0.5:0.5:2.5; %Vertial scalings for airfoil to study thickness effects
ang_Vec = -180:2.5:180;%-20:1:20;      %Angles of Attack to study airfoil lift/drag at different angles


for jj=1:length(scale_Vec) %Loops over all vertical scalings
    
    scale = scale_Vec(jj);
    y = Y*scale; % Scale the airfoil vertically
    
    fprintf('Vertical Geometry Scale = %1.02f\n',scale);
    fprintf('Angle of Attack Performed: ');
    
    %Initialize storage for LIFT/DRAG coeffs, and pressure coeffs.
    if jj==1
       c_l = zeros(length(ang_Vec),length(scale_Vec)); 
       c_d = c_l;                                      
       cp = zeros(length(x)-2,length(ang_Vec),length(scale_Vec));
    end
    
    for kk=1:length(ang_Vec) %Loops over all angles of attack
        
        alpha = ang_Vec(kk); %Choose Angle of Attack
        
        fprintf(' %1.0fdeg',alpha);
        n=length(x)-2; 

        % Setup INFLUENCE Matrix, A
        A = please_Give_Influence_Matrix(x,y,n);
 
        % Setup RHS of System of Equations: UX*y_i + UY*x_i    
        [rhs,alpha] = please_Give_RHS_Vector(x,y,alpha,n);

        % Solve for source strengths
        gamma = A\rhs; 

        % Compute/Plot surface pressure coefficient over body via Bernoulli equation
        for i = 1:n
            %i: loops over ALL pts on specific airfoil
            %kk: for a specific ANGLE OF ATTACK
            %%jj: for a specific vertical SCALING
            cp(i,kk,jj) = 1.0 - gamma(i)^2; %Surface Pressure coeff. via Bernoulli eqn.
        end

        % Compute Lift/Drag force coefficient
        [c_l,c_d] = please_Compute_Lift_Drag_Coeffs(kk,jj,x,y,alpha,n,cp(:,kk,jj),c_l,c_d);

    end %ENDS ANGLE-OF-ATTACK LOOP
    
    fprintf('\n\n*************************************************************************\n\n');

    
end %ENDS SCALING LOOP (for vertical scaling of airfoil)


%
% PLOT DATA OF INTEREST
%
ang = -25;
please_Plot_The_Airfoil(ang,x,Y,scale_Vec,w); % Plot the airfoil for angle of attack 0 and ang. 

please_Plot_Pressure_Distributions(x,n,cp); % Plot the pressure distribution over the airfoil

please_Plot_Lift_Drag(ang_Vec,c_l,c_d);     % Plot the Lift and Drag coefficient



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% COMPUTE the INFLUENCE Matrix, A
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = please_Give_Influence_Matrix(x,y,n)

    A=zeros(n+1,n+1);           %Initialize Influence-Matrix, A
    ds=zeros(1,n);              %Initialize Eucliean distance spacing vector (between each pt.)
    for i = 1:n                 %Calculate spacing between adjacent points
        dx= x(i+1)-x(i);
        dy = y(i+1)-y(i);
        ds(i) = sqrt(dx^2+dy^2);%Euclidean distance between adjoining points
    end

    for j = 1:n          %Loops over all the structure points
    
        A(j,n+1) = 1.0;
    
        for i = 1:n      %Loops over all the points on the structure (to compare pt. considered to others)
        
            if i == j    %Both loops pointing at same point
        
                A(i,i) = ds(i)/(2.*pi) *(log(0.5*ds(i)) - 1.0); %Sets source pt. singularity 
        
            else

                %Center of interval
                xj_C = ( x(j) + x(j+1) )/2;
                yj_C = ( y(j) + y(j+1) )/2;
            
                %j: panel being considering
                %i: influencing panel, j.
            
                %LINE INTEGRAL FORM
                a = xj_C - x(i);
                b = x(i+1) - x(i);
            
                c = yj_C - y(i);
                d = y(i+1) - y(i);
            
                p1 = -2*(b^2+d^2);
                p2 = 2*(b*c-a*d)*( atan2( (b*c-a*d) , (a*b+c*d-b^2-d^2) ) - atan2( (b*c-a*d) , (a*b+c*d) ) );
                p3 = (a*b+c*d)*log(a^2+c^2);
                p4 = (-a*b-c*d+b^2+d^2)*log( (a-b)^2 + (c-d)^2 );
                int_factor = sqrt( b^2 + d^2 ); %int. factor, i.e., ds = ||r'(t)|| dt in line integral formulation
            
                A(j,i) = 1/(4*pi) * 1/(b^2+d^2) * (p1+p2+p3+p4) * int_factor;

            end     
        end
        
        A(n+1,1) = 1.0;
        A(n+1,n) = 1.0;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% COMPUTE the RHS of the System of Equations
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [rhs,alpha] = please_Give_RHS_Vector(x,y,alpha,n)

% Assemble the Right hand Side of the Matrix system
rhs=zeros(n+1,1);        %Initialize RHS-vector
alpha = alpha * pi /180; %Convert angle to radians
xmid=zeros(n,1);         %Initialize middle x-pt vector
ymid=zeros(n,1);         %Initialize middle y-pt vector
for i = 1:n
    xmid(i,1) = (x(i) + x(i+1))/2;  %Define x-pt in middle of each panel
    ymid(i,1) = (y(i) + y(i+1))/2;  %Define y-pt in middle of each panel
  
    rhs(i,1) = ymid(i,1) * cos(alpha) - xmid(i) * sin(alpha);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% COMPUTE the LIFT/DRAG coefficients
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [c_l,c_d] = please_Compute_Lift_Drag_Coeffs(kk,jj,x,y,alpha,n,cp,c_l,c_d)

%
% Compute Lift and Drag Coefficients
%
c_Fx = 0.0; %initialize for integration
c_Fy = 0.0; %initialize for integration

% We assume that the airfoil has unit chord
% we assume that the leading edge is at i = nl; <---ASSUME!!!
for i=1:n

    dx = x(i+1) - x(i); %horizontal length of each panel
    dy = y(i+1) - y(i); %vertical length of each panel
    
    %Integrating pressure coeff., cp, over each panel in both x and y directions
    %note: does riemann integrals
    c_Fy = c_Fy - cp(i)* dx; 
    c_Fx = c_Fx + cp(i)* dy;

end

%
%Compute LIFT/DRAG coefficients
%Lift Coeff = (Vert.Force.Coeff)*cos(alpha) - (Horiz.Force.Coeff)*sin(alpha)
%Drag Coeff = (Vert.Force.Coeff)*sin(alpha) + (Horiz.Force.Coeff)*cos(alpha)
%
c_l(kk,jj) = c_Fy * cos(alpha) - c_Fx * sin(alpha);
c_d(kk,jj) = c_Fy * sin(alpha) + c_Fx * cos(alpha);

%fprintf('\nAngle of Attack: %d deg.\n',alpha*180/pi);
%fprintf('Lift Coeff: %d\n',c_l(kk,jj));
%fprintf('Drag Coeff: %d\n\n',c_d(kk,jj));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PLOT the Airfoil for angle of attack of zero, as well as the input, ang.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function please_Plot_The_Airfoil(ang,x,Y,sVec,w)

% ang: angle to rotate geometry by
% x: x-Values on airfoil
% Y: y-Values on airfoil
% sVec: vector of vertical scalings

figure(1)
plot(x,Y,'o-'); hold on;
X = x;
x = x + w/2;
xR = x.*cos(ang*pi/180) - Y.*sin(ang*pi/180); %Rotate Airfoil x-Vals by ang deg.
yR = x.*sin(ang*pi/180) + Y.*cos(ang*pi/180); %Rotate Airfoil y-Vals by ang deg.
xR = xR - w/2;
plot(xR,yR,'ko-');
xlabel('x');
ylabel('y');
legend('Angle Of Attack: 0 deg','Angle of Attack: 25 deg');
title('Airfoil Geometry');
axis square;

figure(2)
plot(X,sVec(1)*Y,'r*-'); hold on;
plot(X,sVec(2)*Y,'g*-'); hold on;
plot(X,sVec(3)*Y,'*-'); hold on;
plot(X,sVec(4)*Y,'k*-'); hold on;
plot(X,sVec(5)*Y,'m*-'); hold on;
xlabel('x');
ylabel('y');
legend(num2str(sVec(1)),num2str(sVec(2)),num2str(sVec(3)),num2str(sVec(4)),num2str(sVec(5)));
title('Airfoil Geometry');
axis([-1.2*w/2 1.05*1.5*w 1.1*min(sVec(5)*Y) 1.05*max(sVec(5)*Y)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PLOT the Pressure Distributions for Various Scalings and Angles
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function please_Plot_Pressure_Distributions(x,n,cp)

xmid = zeros(n);
for i=1:n
    xmid(i) = (x(i) + x(i+1))/2;  %Define x-pt in middle of each panel
end

figure(3)
subplot(1,2,1)
plot(xmid,cp(:,10,5),'.-'); hold on;
plot(xmid,cp(:,20,5),'r.-'); hold on;
plot(xmid,cp(:,30,5),'m.-'); hold on;
legend('AOA: -90','AOA: -45','AOA: 0');
xlabel('x');
ylabel('Surface Pressure Coeff., cp');
title('Angle of Attack Effects');
%
subplot(1,2,2)
plot(xmid,cp(:,20,1),'.-'); hold on;
plot(xmid,cp(:,20,3),'r.-'); hold on;
plot(xmid,cp(:,20,5),'m.-'); hold on;
legend('Scale: 0.1','Scale: 0.3','Scale: 0.5');
xlabel('x');
ylabel('Surface Pressure Coeff., cp');
title('Vertical Scaling Effects');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PLOT the LIFT/DRAG coefficients as a function of ANGLE OF ATTACK for
% various vertical geometric scalings.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function please_Plot_Lift_Drag(ang_Vec,c_l,c_d)

figure(4)
subplot(1,2,1)
plot(ang_Vec,c_l(:,1),'r*-'); hold on;
plot(ang_Vec,c_l(:,2),'g*-'); hold on;
plot(ang_Vec,c_l(:,3),'*-'); hold on;
plot(ang_Vec,c_l(:,4),'k*-'); hold on;
plot(ang_Vec,c_l(:,5),'m*-'); hold on;
plot(ang_Vec,c_l(:,4),'k*-'); hold on; %Below to make plot look better.
plot(ang_Vec,c_l(:,3),'*-'); hold on;
plot(ang_Vec,c_l(:,2),'g*-'); hold on;
plot(ang_Vec,c_l(:,1),'r*-'); hold on;
xlabel('Angle of Attack');
ylabel('LIFT Coefficient');
title('LIFT vs. AOA');
legend('scale=0.1','scale=0.2','scale=0.3','scale=0.4','scale=0.5');
%
subplot(1,2,2)
plot(ang_Vec,c_d(:,1),'r*-'); hold on;
plot(ang_Vec,c_d(:,2),'g*-'); hold on;
plot(ang_Vec,c_d(:,3),'*-'); hold on;
plot(ang_Vec,c_d(:,4),'k*-'); hold on;
plot(ang_Vec,c_d(:,5),'m*-'); hold on; %Below to make plot look better.
plot(ang_Vec,c_d(:,4),'k*-'); hold on; %Below to make plot look better.
plot(ang_Vec,c_d(:,3),'*-'); hold on;
plot(ang_Vec,c_d(:,2),'g*-'); hold on;
plot(ang_Vec,c_d(:,1),'r*-'); hold on;
xlabel('Angle of Attack');
ylabel('DRAG Coefficient');
title('DRAG vs. AOA');
legend('scale=0.1','scale=0.2','scale=0.3','scale=0.4','scale=0.5');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Read in INPUT data
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%function [x,y] = please_Read_In_Data()
%
%  Open a File and read airfoil coordinates
%
%fileID = fopen('input_data.txt','r');
%
%fclose(fileID);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: returns AIRFOIL geometry (x,y)-values centered at (w/2,0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xAF,yAF] = please_Give_Me_Airfoil_Geometry(w,ds)

% Airfoil Design based on JOUKOVSKY AIRFOIL
% w: "length" of airfoil

% 1st: Make A Circle Centered at the Origin
r=1.2;   %radii of original circle before doing Joukovsky
[xC,yC] = make_Circle_Geometry(r,ds);
xC = xC - 0.1;
yC = yC + 0.45;

% 2nd: Convert Values to A Vector of Complex Numbers
z = xC + 1i*yC;

% 3rd: Use Joukowski Transform: A = z + 1/z
A = z + 1./z;

% 4th: Separate Real and Imag. Parts (real are 'new' x's, imag are 'new' y's)
xAF = real(A);
yAF = imag(A);

% 5th: Scale to get desired width
xAF = 0.5*w*xAF + w/2;
yAF = 0.5*w*yAF;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: returns CIRCLE geometry (x,y)-values centered at (0,0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xC,yC] = make_Circle_Geometry(r,ds)

%r: radii of circle
%ds: Lagrangian spacing

dANG = ds/r;   %Incremental change in angle between points (used s = r*ANG)
ANG = 0:dANG:2*pi;

for i=1:length(ANG)
   xC(i) = r*cos(ANG(i));
   yC(i) = r*sin(ANG(i));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: prints simulation information to screen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function please_Print_Sim_Info()

fprintf('\n-------------------------------------------------------------------------\n\n');
fprintf('  Panel Method: Here used to solve the incompressible potential flow \n');
fprintf('                equations in 2D.\n\n');

fprintf('  Author: Nicholas A. Battista\n');
fprintf('  Created: January 6, 2015\n');
fprintf('  Modified: February 5, 2015\n\n');

fprintf('-------------------------------------------------------------------------\n\n');
fprintf('  Assumptions for Incompressible Potential Flow\n\n');
fprintf('                1. Inviscid\n');
fprintf('                2. Incompressible div(V) = 0\n');
fprintf('                3. Irrotational   curl(V) = 0\n');
fprintf('                4. Steady         partial(u)/partial(t) = 0\n\n');

fprintf('-------------------------------------------------------------------------\n\n');
fprintf('  What it does: \n');
fprintf('                This method finds the lift and drag coefficients around an\n');
fprintf('                airfoil shape, chosen by the user. It also computes the\n');
fprintf('                pressure distribution over the airfoil as well.\n\n');
fprintf('\n*************************************************************************\n\n');
fprintf('                     -----> SIMULATION BEGIN <------');
fprintf('\n\n*************************************************************************\n\n');


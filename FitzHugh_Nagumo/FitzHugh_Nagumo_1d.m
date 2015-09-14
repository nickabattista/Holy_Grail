function FitzHugh_Nagumo_1d()

% This script solves the FitzHugh-Nagumo Equations in 1d, which are 
% a simplified version of the more complicated Hodgkin-Huxley Equations. 
%
% Author:  Nick Battista
% Created: 09/11/2015
%
% Equations:
% dv/dt = D*grad(v) + v*(v-a)*(v-1) - w - I(t)
% dw/dt = eps*(v-gamma*w)
%
% Variables & Parameters:
% v(x,t): membrane potential
% w(x,t): blocking mechanism
% D:      diffusion rate of potential
% a:      threshold potential
% gamma:  resetting rate
% eps:    strength of blocking
% I(t):   initial condition

% Parameters in model (* based off Baird 2014 for Ascidian Heart Tube *) %
D = 315.5;        % Diffusion coefficient
a = 0.5;        % Threshold potential
gamma = 0.5;    % Resetting rate
eps = 0.1;      % Blocking strength

% Discretization/Simulation Parameters %
N = 2000;       % # of discretized points
L = 500;        % Length of domain, [0,L]
dx = L/N;       % Spatial Step
x = 0:dx:L;     % Computational Domain
dt = 1e-4;      % Time-step
T_final = 1.0;  % Final time for simulation


% Initialization %
v = zeros(1,N+1);
w = v;
t=0;
tVec = 0:dt:T_final;
Nsteps = length(tVec);
vNext = zeros(Nsteps,N+1); vNext(1,:) = v;
wNext = zeros(Nsteps,N+1); wNext(1,:) = w;

%
% **** % **** BEGIN SIMULATION! **** % **** %
%
for i=2:Nsteps;
    t = t+dt;                         % Update the time
    
    DD_v_p = give_Me_Laplacian(v,dx); % Give Laplacian 
    
    % Update potential and blocking mechanism, using Forward Euler
    vN = v + dt* ( D*DD_v_p - v.*(v-a).*(v-1) - w - Iapp(dx,x,t) );
    wN = w + dt* ( eps*(v - gamma*w) );
    
    % Update time-steps
    v = vN;
    w = wN;
    
    % Store time-step values
    vNext(i,:) = v;
    wNext(i,:) = w;
    
    % Plot
    if rem(i,50) == 0
    plot(x,v,'*-'); hold on;
    axis([0 L -0.03 0.01]);
    minV = min(v);
    strTitle = sprintf('time: %f,  min V: %f',t,minV);
    title(strTitle);
    pause(0.05)
    clf;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: the injection function, Iapp = Iapp(x,t)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = Iapp(dx,x,t)

N = length(x);
xM = x(floor(length(x)/2) );

val = zeros(1,N);

tt = rem(t,0.25);

if ( ( tt > 0.1 ) && ( tt < 0.15 ) )
    %val = 0.1*abs( sin(2*pi*t) ).*exp( -(x - xM).^2  ./ (2*0.1) );
    %val = 0.1.*exp( -(x - xM).^2  ./ (2*0.1) );
    val(950:1050) = 0.5;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FUNCTION: gives Laplacian of the membrane potential, note: assumes
% periodicity and uses the 2nd order central differencing operator.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DD_v = give_Me_Laplacian(v,dx)

Npts = length(v);
DD_v = zeros(1,Npts);

for i=1:Npts
   if i==1
       DD_v(i) = ( v(i+1) - 2*v(i) + v(end) ) / dx^2;
   elseif i == Npts
       DD_v(i) = ( v(1) - 2*v(i) + v(i-1) ) / dx^2;
   else
       DD_v(i) = ( v(i+1) - 2*v(i) +  v(i-1) ) /dx^2;
   end

end


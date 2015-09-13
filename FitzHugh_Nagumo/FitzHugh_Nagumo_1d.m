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
D = 100;        % Diffusion coefficient
a = 0.1;        % Threshold potential
gamma = 0.5;    % Resetting rate
eps = 0.1;      % Blocking strength
Ival = 0.5;     % Current Injection

% Discretization/Simulation Parameters %
N = 1000;       % # of discretized points
L = 500;        % Length of domain, [0,L]
dx = L/N;       % Spatial Step
x = 0:dx:L;     % Computational Domain
dt = 1e-2;      % Time-step
T_final = 1.0;  % Final time for simulation


% Initialization %
v = zeros(1,N+1);
w = v;
tVec = 0:dt:T_final;
Nsteps = length(tVec);
vNext = zeros(Nsteps,N+1); vNext(1,:) = v;
wNext = zeros(Nsteps,N+1); wNext(1,:) = w;

%
% **** % **** BEGIN SIMULATION! **** % **** %
%
for i=2:Nsteps;
    
    DD_v_p = give_Me_Laplacian(v,dx);
    
    % Update potential and blocking mechanism, using Forward Euler
    v = v + dt* ( D*DD_v_p + v.*(v-a).*(v-1) - w - Ival );
    w = w + dt* ( eps*(v - gamma*w) );
    
    % Store time-step values
    vNext(i,:) = v;
    wNext(i,:) = w;
    
    % Plot
    plot(x,v,'*-'); hold on;
    pause(0.5);
    clf;
    
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


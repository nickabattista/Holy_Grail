
vidobj = VideoWriter('FHNagumo2d_movie','MPEG-4');

k=1;

L=500;
nX=100;
dx=L/nX;
nT=3000;
T=1000;
dt=T/nT;

D=10;
a=.05;
b = .01; %parameters in fitzhugh -- epsilon, originally 0.01
gamma=1; %parmater in fitzhugh -- originally 1

N=nX;
v=zeros(nX);
v_old=v;
w=v;
w_old=w;


t=0;
n=0;

I_appl=zeros(nT,4);
I_appl(1500:3000,1)=.1;
I_appl(1000:1100,2)=.1;
I_appl(100:200,3)=.1;
I_appl(1000:1100,4)=.1;
open(vidobj);
figure(1)
surf(v)
axis([0 100 0 100 -0.3 1.2 0 1])
record=0;
I_area=zeros(size(v));

while t<T
    I_area(1:5,1:5)=I_appl(n+1,1);
    I_area(1:5,26:30)=I_appl(n+1,2);
    I_area(1:5,51:55)=I_appl(n+1,3);
    I_area(1:5,76:80)=I_appl(n+1,4);
    
    
    for i=2:N-1
        for j=2:N-1
            
            v(i,j) = v_old(i,j)+dt*(((D/(dx*dx))*(v_old(i+1,j)+v_old(i-1,j)+v_old(i,j+1)+v_old(i,j-1)-4*v_old(i,j)))-v_old(i,j)*(v_old(i,j)-1)*(v_old(i,j)-a)-w_old(i,j)+I_area(i,j));
            
        end
    end
    
    %     %Periodic
    %     v(1,2:N-1) = v_old(1,2:N-1)+dt.*(((D/(dx*dx)).*(v_old(2,2:N-1)+v_old(N,2:N-1)+v_old(1,1:N-2)+v_old(1,3:N)-4.*v_old(1,2:N-1)))-v_old(1,2:N-1).*(v_old(1,2:N-1)-1).*(v_old(1,2:N-1)-a)-w_old(1,2:N-1)+I_area(1,2:N-1));
    %     v(N,2:N-1) = v_old(N,2:N-1)+dt.*(((D/(dx*dx)).*(v_old(1,2:N-1)+v_old(N-1,2:N-1)+v_old(N,1:N-2)+v_old(N,3:N)-4.*v_old(N,2:N-1)))-v_old(N,2:N-1).*(v_old(N,2:N-1)-1).*(v_old(N,2:N-1)-a)-w_old(N,2:N-1)+I_area(N,2:N-1));
    %     v(2:N-1,1) = v_old(2:N-1,1)+dt.*(((D/(dx*dx)).*(v_old(1:N-2,1)+v_old(3:N,1)+v_old(2:N-1,N)+v_old(2:N-1,2)-4.*v_old(2:N-1,1)))-v_old(2:N-1,1).*(v_old(2:N-1,1)-1).*(v_old(2:N-1,1)-a)-w_old(2:N-1,1)+I_area(2:N-1,1));
    %     v(2:N-1,N) = v_old(2:N-1,N)+dt.*(((D/(dx*dx)).*(v_old(1:N-2,N)+v_old(3:N,N)+v_old(2:N-1,N-1)+v_old(2:N-1,1)-4.*v_old(2:N-1,N)))-v_old(2:N-1,N).*(v_old(2:N-1,N)-1).*(v_old(2:N-1,N)-a)-w_old(2:N-1,N)+I_area(2:N-1,N));
    %
    %     v(1,1)=v_old(1,1)+dt*(((D/(dx*dx)).*(v_old(N,1)+v_old(2,1)+v_old(1,N)+v_old(1,2)-4*v_old(1,1)))-v_old(1,1).*(v_old(1,1)-1).*(v_old(1,1)-a)-w_old(1,1)+I_area(1,1));
    %     v(1,N)=v_old(1,N)+dt*(((D/(dx*dx)).*(v_old(N,N)+v_old(2,N)+v_old(1,N-1)+v_old(1,1)-4*v_old(1,N)))-v_old(1,N)*(v_old(1,N)-1)*(v_old(1,N)-a)-w_old(1,N)+I_area(1,N));
    %     v(N,1)=v_old(N,1)+dt*(((D/(dx*dx)).*(v_old(N-1,1)+v_old(1,1)+v_old(N,N)+v_old(N,2)-4*v_old(N,1)))-v_old(N,1)*(v_old(N,1)-1)*(v_old(N,1)-a)-w_old(N,1)+I_area(N,1));
    %     v(N,N)=v_old(N,N)+dt*(((D/(dx*dx)).*(v_old(N-1,N)+v_old(1,N)+v_old(N,N-1)+v_old(N,1)-4*v_old(N,N)))-v_old(N,N)*(v_old(N,N)-1)*(v_old(N,N)-a)-w_old(N,N)+I_area(N,N));
    
    
    v(1,2:N-1) = v_old(1,2:N-1)+dt.*(((D/(dx*dx)).*(v_old(2,2:N-1)+v_old(1,1:N-2)+v_old(1,3:N)-3.*v_old(1,2:N-1)))-v_old(1,2:N-1).*(v_old(1,2:N-1)-1).*(v_old(1,2:N-1)-a)-w_old(1,2:N-1)+I_area(1,2:N-1));
    v(N,2:N-1) = v_old(N,2:N-1)+dt.*(((D/(dx*dx)).*(v_old(N-1,2:N-1)+v_old(N,1:N-2)+v_old(N,3:N)-3.*v_old(N,2:N-1)))-v_old(N,2:N-1).*(v_old(N,2:N-1)-1).*(v_old(N,2:N-1)-a)-w_old(N,2:N-1)+I_area(N,2:N-1));
    v(2:N-1,1) = v_old(2:N-1,1)+dt.*(((D/(dx*dx)).*(v_old(1:N-2,1)+v_old(3:N,1)+v_old(2:N-1,N)+v_old(2:N-1,2)-4.*v_old(2:N-1,1)))-v_old(2:N-1,1).*(v_old(2:N-1,1)-1).*(v_old(2:N-1,1)-a)-w_old(2:N-1,1)+I_area(2:N-1,1));
    v(2:N-1,N) = v_old(2:N-1,N)+dt.*(((D/(dx*dx)).*(v_old(1:N-2,N)+v_old(3:N,N)+v_old(2:N-1,N-1)+v_old(2:N-1,1)-4.*v_old(2:N-1,N)))-v_old(2:N-1,N).*(v_old(2:N-1,N)-1).*(v_old(2:N-1,N)-a)-w_old(2:N-1,N)+I_area(2:N-1,N));
    
    v(1,1)=v_old(1,1)+dt*(((D/(dx*dx)).*(v_old(2,1)+v_old(1,N)+v_old(1,2)-3*v_old(1,1)))-v_old(1,1).*(v_old(1,1)-1).*(v_old(1,1)-a)-w_old(1,1)+I_area(1,1));
    v(1,N)=v_old(1,N)+dt*(((D/(dx*dx)).*(v_old(2,N)+v_old(1,N-1)+v_old(1,1)-3*v_old(1,N)))-v_old(1,N)*(v_old(1,N)-1)*(v_old(1,N)-a)-w_old(1,N)+I_area(1,N));
    v(N,1)=v_old(N,1)+dt*(((D/(dx*dx)).*(v_old(N-1,1)+v_old(N,N)+v_old(N,2)-3*v_old(N,1)))-v_old(N,1)*(v_old(N,1)-1)*(v_old(N,1)-a)-w_old(N,1)+I_area(N,1));
    v(N,N)=v_old(N,N)+dt*(((D/(dx*dx)).*(v_old(N-1,N)+v_old(N,N-1)+v_old(N,1)-3*v_old(N,N)))-v_old(N,N)*(v_old(N,N)-1)*(v_old(N,N)-a)-w_old(N,N)+I_area(N,N));
    
    
    for i=1:N
        for j=1:N
            w(i,j) = w_old(i,j)+dt*(b*(v_old(i,j)-gamma*w_old(i,j)));
        end
    end
    
    if (t>record)
        figure(1)
        surf(v)
        axis([0 100 0 100 -0.3 1.2 0 1])
        title(['Step = ',num2str(n)])
        record=record+dt*10;
        F(k) = getframe;    %add frame to the movie
        frame = getframe;
        
        writeVideo(vidobj, frame);
        
        k = k+1;
        
    end
    
    t=t+dt;
    n=n+1
    
    v_old=v;
    w_old=w;
    
end
close(vidobj);

vidobj = VideoWriter('FHNagumo1d_movie','MPEG-4');

k=1;

L=500;
nX=100;
dx=L/nX;
nT=3000;
T=1000;
dt=T/nT;

D=20;
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
I_appl(100:200,3)=.025;
I_appl(1000:1100,4)=.1;
plot(I_appl)
pause(); 
open(vidobj);
figure(1)
plot(v)
hold on
plot(w,'-r')
%plot(I_appl)
hold off
axis([0 100 -0.3 1.1])

record=0;
I_area=zeros(size(v));

while t<300
    I_area(1:5)=I_appl(n+1,1);
    I_area(26:30)=I_appl(n+1,2);
    I_area(51:55)=I_appl(n+1,3);
    I_area(76:80)=I_appl(n+1,4);
    
    plot(I_area)
    pause(0.05);
    
    for i=2:N-1
        
        
        v(i) = v_old(i)+dt*( ((D/(dx*dx))*( v_old(i+1)+v_old(i-1)-2*v_old(i) )) -v_old(i)*(v_old(i)-1)*(v_old(i)-a)-w_old(i)+I_area(i) );
    end

    v(1)=v_old(1)+dt*(((D/(dx*dx)).*(v_old(N)+v_old(2)-2*v_old(1)))-v_old(1).*(v_old(1)-1).*(v_old(1)-a)-w_old(1)+I_area(i));
    v(N)=v_old(N)+dt*(((D/(dx*dx)).*(v_old(N-1)+v_old(1)-2*v_old(N)))-v_old(N)*(v_old(N)-1)*(v_old(N)-a)-w_old(N)+I_area(i));
    
    for i=1:N
        
        w(i) = w_old(i)+dt*(b*(v_old(i)-gamma*w_old(i)));
        
    end
    
%     if (t>record)
%         figure(1)
%         
%         plot(v)
%         hold on
%         plot(w,'-r')
%         %plot(I_appl)
%         hold off
%         axis([0 100 -0.3 1.1])
%         record=record+dt*10;
%         F(k) = getframe;    %add frame to the movie
%         frame = getframe;
%         title(['Step = ',num2str(n)])
%         writeVideo(vidobj, frame);
%         
%         k = k+1;
%         
%     end
%     
    t=t+dt;
    n=n+1
    
    v_old=v;
    w_old=w;
    
end
close(vidobj);
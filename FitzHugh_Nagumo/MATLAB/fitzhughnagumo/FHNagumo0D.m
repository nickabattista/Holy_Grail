clear all
close all


nT=100000;
T=1000;
dt=T/nT;

D=20;
a=.1;
b = .01; %parameters in fitzhugh -- epsilon, originally 0.01
gamma=.01; %parmater in fitzhugh -- originally 1

v=zeros(nT+1,1);
w=v;


t(1)=0;


I_appl=zeros(1,nT+1);
I_appl(5000:6000)=.1;
I_appl(25000:26000)=.1;

I_appl(45000:46000)=.1;
I_appl(65000:66000)=.1;
I_appl(85000:86000)=.1;


I_area=zeros(size(v));
for n=1:nT
    t(n+1)=t(n)+dt;

    v(n+1) = v(n)+dt*(-v(n)*(v(n)-1)*(v(n)-a)-w(n)+I_appl(n));
    
    w(n+1) = w(n)+dt*(b*(v(n)-gamma*w(n)));

end

figure(1)
plot(t,v);
hold on
plot(t,w,'-r');
plot(t,I_appl,'-g')
hold off
legend('V','W', 'I_{A}');
axis([0 1000 -.5 1.2])

y=linspace(-0.05,.2,20);
x=linspace(-0.4,1.2,20);
[Xmesh,Ymesh]=meshgrid(x,y);

U=-Xmesh.*(Xmesh-1).*(Xmesh-a)-Ymesh;
V=b*Xmesh-gamma*Ymesh;

wnull=(b/gamma)*x;
vnull=-x.*(x-1).*(x-a);

figure(2)
plot(v,w,'-k');
hold on
quiver(Xmesh,Ymesh,V,U,'b')
plot(x,wnull,'r')
plot(x,vnull,'g')
axis([-.4 1.2 -.05 .2])
hold off
xlabel('V')
ylabel('W')


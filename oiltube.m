clear;
n_ele = 100;
dt = 0.001;
timestep = 15000;
dx = 500/n_ele;
p = ones(n_ele,1);
v = ones(n_ele+1,1);
rho = ones(n_ele,1);

%initialize
p = 150.*p;
v = 10.*ones(n_ele+1,1);
rho = 0.0004.*p+0.8;
rho0 = 0.85;

drho = ones(n_ele,1);
dv = ones(n_ele+1,1);
v(1) = 0; v(n_ele+1) = 0;
%update alg
rho_record = zeros(timestep,n_ele);
for step = 1:timestep
    %boundary condition
    v(1) = 2*sqrt((180-p(1))/rho0);
    v(n_ele+1) = 2*sqrt(p(n_ele)/rho(n_ele));
       
    %coef_out : the radius of jet
    coef_out = 0.1;
    coef_in = 0.6;
    drho(1) = (rho0.*v(1)*coef_in - rho(1).*v(2))*dt/dx;
    drho(n_ele) = (rho(n_ele-1).*v(n_ele)- v(n_ele+1).*rho(n_ele)*coef_out)*dt/dx;
    dv(1) = 0;
    dv(n_ele+1) = 0;
    %inner loop
    for i = 2:n_ele-1
        u1 = rho(i-1)*(v(i) > 0) + rho(i)*(v(i) < 0);
        u2 = rho(i)*(v(i+1) > 0) + rho(i+1)*(v(i+1) < 0);
        drho(i) = (u1.*v(i) - u2.*v(i+1))*dt/dx;
        dv(i) = (p(i-1)+rho(i-1).*((v(i-1) + v(i)).^2)./4) - (p(i)+rho(i).*((v(i+1) + v(i)).^2)./4);
        dv(i) = dv(i)*2/(rho(i-1)+rho(i))*dt/dx;
    end
    i = n_ele;
    dv(i) = (p(i-1)+rho(i-1).*((v(i-1) + v(i)).^2)./4) - (p(i)+rho(i).*((v(i+1) + v(i)).^2)./4);
    dv(i) = dv(i)*2/(rho(i-1)+rho(i))*dt/dx;
    %update 
    rho = rho + drho;
    v = v + dv;
    for i = 1:n_ele
           p(i) = 2500*(rho(i)- 0.8);
    end
    
end
plot(rho)

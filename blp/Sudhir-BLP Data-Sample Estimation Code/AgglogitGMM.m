% Sample code to illustrate estimation of BLP random coeffients model 
% Written by K. Sudhir, Yale SOM
% For  Quantitative Marketing & Structural Econometrics Workshop @Duke-2013
% This is code for the GMM objective function that needs to minimized
function f=AgglogitGMM(b)
global w1 w2 wp x y z NCons NObs NObs1 xlin blin W z s blin log_s_s0;
%Step 1: Multiplying fixed Standard Normal draws  by lower triangular Cholesky matrix
% parameters to get the multivariate heterogeneity draws on intercepts and
% prices
aw=w1;
awp=wp;
aw1=b(1)*w1+b(2)*w2;
aw2=b(3)*w1;
%Step 2: Constructing the nonlinear part of the share equation (mu)
% using the heterogeneity draws on intercepts and price coeff
for i=1:1:size(w1,2);
    aw(:,i)=aw1(:,i).*x(:,1)+aw2(:,i).*x(:,2);
    awp(:,i)=b(4)*x(:,3).*wp(:,i);
end;
delta=log_s_s0;
Err=100;
Tol=1e-12;
de1=delta;
% Step 3: Contraction Mapping to get the delta_jt until Tolerance level met
while (Err >= Tol)
    de=de1;
    sh=zeros(NObs1,1);
    psh=zeros(NObs1,1);
    %Obtaining the predicted shares based on model
    for i=1:1:NCons;
        psh=exp(aw(:,i)+awp(:,i)+de);
        psh=reshape(psh',2,NObs)';
        spsh=sum(psh')';
        psh(:,1)=psh(:,1)./(1+spsh);
        psh(:,2)=psh(:,2)./(1+spsh);
        sh=sh+reshape(psh',NObs1,1);
    end;
    sh=sh/NCons;
    %Updating delta_jt based on difference between actual share and 
    %predicted shares
    de1=de+log(s)-log(sh);
    Err=max(abs(de1-de));
end;
delta=de1;
% Step 4: Getting the linear parameters and setting up the objective fn
blin=inv(xlin'*z*W*z'*xlin)*(xlin'*z*W*z'*delta);
ksi=delta-xlin*blin;
% The GMM objective function that will be optimized over to get 
% nonlinear parameters
f=ksi'*z*W*z'*ksi;
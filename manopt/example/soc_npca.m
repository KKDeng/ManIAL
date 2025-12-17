function [X_Soc, F_SOC,sparsity_soc,time_soc,error_XPQ, iter_soc, flag_succ]=soc_npca(B,option,problem)
%min -Tr(X'*A*X)+ mu*norm(X,1) s.t. X'*X=Ir.
% A = B'*B type = 0 or A = B  type = 1
tic; prox_g = problem.prox_g;
r = option.r;
n = option.n;
mu = option.mu;
maxiter =option.maxiter;
tol = option.tol;
type = option.type;
if type==0 % data matrix
    A = -B'*B;
else
    A = -B;
end

rho = 2* svds(B,1)^2  ;%  n/30 not converge   1.9* sometimes not converge
lambda = rho;
P = option.phi_init;    Q = P;
Z = zeros(n,r); 
b=Z;  
F_ad=zeros(maxiter,1);

Ainv = inv( 2*A + (rho+lambda)*eye(n));

for itera=1:maxiter
    LZ = rho*(P-Z)+lambda*(Q-b);
    X = Ainv*LZ;
    Q = prox_g(X+b,1);
    %%%% solve P
    
    Y = X + Z;
    [U,~,V]= svd(Y,0);
    P = U*V';
    %%%%%%%%%
    Z  = Z+X-P;
    b  = b+X-Q;
    
    if itera>2
        normXQ = norm(X-Q,'fro');
        normQ = norm(Q,'fro');
        normX = norm(X,'fro');
        normP = r;
        normXP = norm(X-P,'fro');
        if  normXQ/max(1,max(normQ,normX)) + normXP/max(1,max(normP,normX)) <tol
            if type == 0 % data matrix
                AP = -(B'*(B*P));
            else
                AP = -(B*P);
            end
            F_ad(itera)= sum(sum(X.*(AP)));
       
            break;
        end
        
    end
    
    
end
P((abs(P)<=1e-5))=0;
X_Soc=P;
time_soc= toc;
error_XPQ = norm(X-P,'fro') + norm(X-Q,'fro');

sparsity_soc= sum(sum(P==0))/(n*r);
    
        
        flag_succ = 1; % success
        F_SOC = F_ad(itera);
        iter_soc = itera;
        % residual_Q = norm(Q'*Q-eye(n),'fro')^2;
        fprintf('Soc:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
        
        print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
        fprintf(1,print_format, itera, F_ad(itera), time_soc, sparsity_soc,  error_XPQ);
end

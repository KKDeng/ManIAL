%function compare_spca
function  demo_compare_ODL()
clear;
close all;
addpath ../misc
addpath ../SSN_subproblem
addpath ../
addpath ../util

save_root = strcat('../results/odl/');
if ~exist(save_root,'dir')
    mkdir(save_root);
end



D_set=[ 30; 50; 100; 300; 400; ]; %dimension

theta_set = [0.3; 0.4; 0.5];



%% problem setting









rng(1000);   test_num = 1;


%% cycle
for id_D = 1:size(D_set,1)        % n  dimension
    for id_theta = 1:size(theta_set,1)
        theta = theta_set(id_theta);
        D = D_set(id_D); %ambient dimension
        basename = ['odl_',num2str(D),'_',num2str(theta)];
        
        ret_mialm = zeros(test_num,9);
        ret_Rsub = zeros(test_num,3);
        ret_madmm = zeros(test_num,9);
        
        
        for test_random = 1:test_num  %times average.
            
            %rng('shuffle');
            
            
            p = 1.5;   % sample complexity (as power of n)
            m = round(10*D^p);    % number of measurements
            Q = randU(D);     % an uniformly random orthogonal matrix
            X = randn(D, m).*(rand(D, m) <= theta);   % iid Bern-Gaussian model
            Y = Q*X;
            Init = orth(randn(D));
            
            
            
            
            
            
            
            
            A = struct();
            A.applyA = @(X) applyA(Y,X);
            A.applyAT = @(y) applyAT(Y,y);
            
            
            f = struct();
            f.cost_grad = @zeros_obj_grad;
            
            
            
            h = struct();
            h.cost = @(X,lambda) lambda*sum(sum(abs(X)));
            h.prox = @(X,nu,lambda) max(abs(X) - nu*lambda,0).* sign(X);
            h.data = {1};
            
            
            %manifold = unitaryfactory(D);
            manifold = stiefelfactory(D,D);
             %manifold.retr = @retraction;
            
                options_mialm.verbosity = 0;
                options_mialm.max_iter = 200; options_mialm.tol = 1e-8*D^2;
                options_mialm.rho = 1.05;     options_mialm.tau = 0.8;    
                options_mialm.nu0 = 10 ;
                options_mialm.gtol0= 1e-2;   options_mialm.gtol_decrease = 0.9;
                options_mialm.X0 = Init;
                options_mialm.maxitersub = 10; options_mialm.extra_iter = 10;
                options_mialm.verbosity = 1;
                
            
            [X_mialm,Z_mialm,out_mialm]= mialm(A, manifold, f, h, options_mialm);
            ret_mialm(test_random,:) = [out_mialm.obj, out_mialm.sparsity, out_mialm.time, out_mialm.iter, out_mialm.sub_iter, ...
                out_mialm.deltak, out_mialm.etaD, out_mialm.etaC, out_mialm.nrmG];
            err = sum( abs( max(abs(X_mialm'*Q),[],2) - ones(D,1) )  );
       
            
            %%%%%% Riemannian subgradient parameter
            option_Rsub.F_mialm = out_mialm.obj;
            option_Rsub.max_iter = 1e4;
            option_Rsub.tol = 1e-6;
            option_Rsub.Q = Q;
            
            [X_Rsub, out_Rsub] = R_sub_ODL(Y, Init, option_Rsub);
            F_Rsub = out_Rsub.obj; norm(Y'*X_Rsub,1)
            ret_Rsub(test_random,:) = [F_Rsub,out_Rsub.time, out_Rsub.iter];
            
            
            
            
            
            options_admm = options_mialm;
            options_admm.mu = 0.5/svds(Y,1)^1;
            options_admm.iter = 5000;  options_admm.opt = out_mialm.obj;
            options_admm.maxiter = 100;
            options_admm.tolgradnorm = 1e-4;
            
            [X_madmm,Z_madmm,out_madmm]=madmm(A, manifold, f, h, options_admm);
            ret_madmm(test_random,:) = [out_madmm.obj, out_madmm.sparsity, out_madmm.time, out_madmm.iter, out_madmm.sub_iter, ...
                out_madmm.deltak, out_madmm.etaD, out_madmm.etaC, out_madmm.nrmG];
            
        end
        
        save_path = strcat(save_root,basename,'.mat');
        save(save_path, 'ret_mialm', 'ret_madmm', 'ret_Rsub', ...
            'X_mialm', 'X_madmm', 'Z_mialm', 'Z_madmm',  'X_Rsub', 'D','theta','p' );
        
    end
end

    function re = applyA(Y,X)
        re = Y'*X;
    end
    function re = applyAT(Y,y)
        re = Y*y;
    end
    function [cost,grad] = zeros_obj_grad(X,lambda)
        cost = 0;
        grad = 0;
    end
end


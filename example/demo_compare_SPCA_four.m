%function compare_spca
function  demo_compare_SPCA_four()
clear;
close all;
addpath ../misc
addpath ../SSN_subproblem
addpath ../
addpath ../util
addpath (genpath('../manopt'));

save_root = strcat('../results/spca/');
if ~exist(save_root,'dir')
    mkdir(save_root);
end


save_root_res = strcat(save_root,'res/');
if ~exist(save_root_res,'dir')
    mkdir(save_root_res);
end

%n_set=[ 200; 300; 500; ]; %dimension
n_set=[ 200; 300; 500; 1000]; %dimension
n_set = [200;500;100];
%format long
r_set = [ 20];   % rank
%r_set = 5;

mu_set = [0.4;0.6;0.8];
%mu_set = [0.5;0.6;0.7;0.8];
%mu_set = 0.5;

%% problem setting



rng(2025);
table_str = '';

%% cycle
for id_n = 1:size(n_set,1)        % n  dimension
    for id_r = 1:size(r_set,1) % r  number of column
        for id_mu = 1:size(mu_set,1)         % mu  sparse parameter
            r = r_set(id_r);
            lambda = mu_set(id_mu);
            n = n_set(id_n);

            basename = [];





            %rng('shuffle');
            m = 5000;
            B = randn(m,n);
            for i=1:m
                B(i,:) = randn(1,n);
            end
            B = B - repmat(mean(B,1),m,1);
            B = normc(B);
            %B = B/sqrt(norm(B'*B,'fro'));

            AtA = B'*B;
            type = 1;

            [phi_init,~] = svd(randn(n,r),0);  % random intialization

            option_Rsub.F_mialm = -1e10;
            option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 5e1;  option_Rsub.tol = 5e-3;
            option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = type;

            [phi_init]= Re_sub_spca(AtA,option_Rsub);


            A = struct();
            A.applyA = @(X) X;
            A.applyAT = @(y) y;

            f = struct();
            f.cost_grad = @pca_cost_grad;
            f.data = {B};

            h = struct();
            h.cost = @(X,lambda) lambda*sum(sum(abs(X)));
            h.prox = @(X,nu,lambda) max(abs(X) - nu*lambda,0).* sign(X);
            h.data = {lambda};


            manifold = stiefelfactory(n,r);




            %%%%%  manpg parameter
            option_manpg.opt = -inf;
            option_manpg.adap = 0;    option_manpg.type = type;
            option_manpg.phi_init = phi_init; option_manpg.maxiter = 20000;  option_manpg.tol =1e-8*n*r;
            option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = lambda;
            option_manpg.inner_iter = 100;

            % profile on
            [X_manpg, F_manpg, sparsity_manpg,time_manpg,...
                maxit_att_manpg, succ_flag_manpg, lins_manpg, in_av_manpg,arrF_manpg,arrtime_manpg]= manpg_orth_sparse(AtA,option_manpg);

            
            
            
                        % manpg with BB step size
                        [X_manpg_BB, F_manpg_BB,sparsity_manpg_BB,time_manpg_BB,...
                            maxit_att_manpg_BB,succ_flag_manpg_BB,lins_adap_manpg,in_av_adap_manpg,arrF_BB,arrtime_BB]= manpg_orth_sparse_adap(AtA,option_manpg);




            options_ial.max_iter = 1000; options_ial.tol = 1e-10*n*r;
            options_ial.rho = 1.8;     options_ial.tau = 0.8;
            options_ial.nu0 = (svds(AtA,1)^1)*2  ;
            %             options_ial.gtol0= 1e-0;   options_ial.gtol_decrease = 0.9;
            options_ial.X0 = phi_init;
            %$%options_ial.maxitersub = 100; options_ial.extra_iter = 100;
            options_ial.verbosity = 1;
            options_ial.obj = F_manpg; options_ial.max_iter = 1000;


            options_ial.flag = 1;
            [X_ial1,z1,out_ial1] = manial(A, manifold, f, h, options_ial);

            time_ial1 = out_ial1.time_arr;
            obj_ial1 = out_ial1.obj_arr - F_manpg;
            iter1 = out_ial1.iter;
            error1 = max(out_ial1.error_arr');

            options_ial.flag = 2; options_ial.rho = 1.5;options_ial.tau = 0.7;
            [X_ial2,z2,out_ial2] = manial(A, manifold, f, h, options_ial);

            time_ial2 = out_ial2.time_arr;
            obj_ial2 = out_ial2.obj_arr - F_manpg;
            iter2 = out_ial2.iter;
            error2 = max(out_ial2.error_arr');




            f = struct();
            f.cost_grad = @partial_pca_cost_grad;
            f.data = {B};


            options_ial3 = options_ial;
            options_ial3.maxitersub = 10;
            options_ial.rho = 1.05;
            options_ial3.batchsize = m/1000;
            options_ial3.max_iter = 100;
            options_ial3.m = m; options_ial3.opt = out_ial1.obj;F_manpg;
            options_ial3.timeout = 200;
            [X_ial3,z3,out_ial3] = stomialm(A, manifold, f, h, options_ial3);

            %             f.cost_grad = @pca_cost_grad2; f.data = {AtA};
            %             options_ial.X0 = X_ial2; options_ial.z = z2;
            %             [X_ial1,z1,out_ial1] = mialm(A, manifold, f, h, options_ial);


            time_ial3 = out_ial3.time_arr;
            obj_ial3 = out_ial3.obj_arr - F_manpg;
            iter3 = out_ial3.iter;
            error3 = max(out_ial3.error_arr');


            %%%%% Riemannian subgradient parameter
            option_Rsub.F_mialm = F_manpg;
            option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 1e2;      option_Rsub.tol = 5e-3;
            option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = type;

             option_Rsub.m = m; option_Rsub.batchsize = m;
             option_Rsub.AX = @(X,sample) partial_pca_AX(X,B,sample);
            [X_Rsub, F_Rsub,sparsity_Rsub,time_Rsub,...
                maxit_att_Rsub,succ_flag_sub,sub_obj,sub_time]= Re_sub_grad_spca_sto(AtA,option_Rsub);
            sub_obj = sub_obj - F_manpg;


            arrF_manpg = arrF_manpg - F_manpg;

            arrF_BB = arrF_BB - F_manpg;

            f0 = figure;

            plot(time_ial1(1:1:iter1),log(obj_ial1(1:1:iter1)),'-.*','LineWidth',1);
            hold on;
            plot(time_ial2(1:1:iter2),log(obj_ial2(1:1:iter2)),'-<','LineWidth',1);
            plot(time_ial3(1:1:iter3+1),log(obj_ial3(1:1:iter3+1)),'-<','LineWidth',1);
            plot(sub_time(1:5:maxit_att_Rsub),log(sub_obj(1:5:maxit_att_Rsub)),'-o','LineWidth',1);
            plot(arrtime_manpg(1:1:maxit_att_manpg),log(arrF_manpg(1:1:maxit_att_manpg)),'-s','LineWidth',1);
            plot(arrtime_BB(1:1:maxit_att_manpg_BB),log(arrF_BB(1:1:maxit_att_manpg_BB)),'--*','LineWidth',1);


            legend('ManIAL-I','ManIAL-II','StoManIAL','Rsub','ManPG','ManPG-ada');
            xlabel('CPU time','FontName','Arial');
            ylabel('$\log  (F(x^k) - F_M)$','interpreter','latex','FontName','Arial');
            name = sprintf('fig//compare_obj-sub-%d-%d-%.1f.png', n,r,lambda);
            saveas(gcf, name,'png');
       %     close(f0);

           
        end
    end
end

disp(newline);
disp(table_str);
save_path = strcat(save_root,'spca_1.txt');
fid = fopen(save_path,'w+');
fprintf(fid,'%s',table_str);


    function [f,g] = pca_cost_grad(X,A)
        BX = A'*(A*X);
        f = -sum(sum(BX.*X));
        g = -2*BX;
    end

    function [f,g] = pca_cost_grad2(X,AtA)
        BX = AtA*X;
        f = -sum(sum(BX.*X));
        g = -2*BX;
    end

    function [f,g] = partial_pca_cost_grad(X,B,sample)
        del = length(sample)/m;
        sampleB = B(sample,:);
        BTBX = sampleB'*(sampleB*X);
        f = -sum(sum(BTBX.*X))*del;
        g = -2*BTBX*del;
    end

   function AX = partial_pca_AX(X,B,sample)
        del = length(sample)/m;
        sampleB = B(sample,:);
        AX = sampleB'*(sampleB*X)*del;
    end


end


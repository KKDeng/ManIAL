function demo_random_cca(test_type)

if nargin<1; test_type = 1; end

isLasso = 1;


addpath ../SCCALab
addpath ../TFOCS-master
addpath ../
addpath ../util
addpath SSN_subproblem
addpath misc
addpath ../scca_penalty
addpath (genpath('../manopt'));
addpath (genpath('SPCA&SCCA'))

save_root = strcat('../results/random/');
if ~exist(save_root,'dir')
    mkdir(save_root);
end


save_root_res = strcat(save_root,'res/');
if ~exist(save_root_res,'dir')
    mkdir(save_root_res);
end

save_root_res1 = strcat(save_root,'res_s/');
if ~exist(save_root_res1,'dir')
    mkdir(save_root_res1);
end


N = 5000; % sample size
p_set = [500]; % dim of X
q_set = [100]; % dim of Y
r_set = [2,3,5];   % column number
lambda_set = [0.1,0.2];
%lambda_set = [0.1,0.2,0.3]/2;
s  = [1:2:20];
ss = [1:4:20];
rng(2025); table_str = '';


for id_p =  1:length(p_set)        % n  dimension
    for id_q = 1 :length(q_set) % r  number of column
        for id_r = 1:length(r_set)  %mu  sparsity parameter
            bestTrace = 10; bestLasso = 10; bestInit = 10; bestPena = 10;
            for id_lambda = 1:length(lambda_set)
                lambda1 = lambda_set(id_lambda);
                lambda2 = lambda_set(id_lambda);
                r = r_set(id_r);
                p = p_set(id_p);
                q = q_set(id_q);



                basename = ['rand_',num2str(test_type),'_',num2str(p),'_',num2str(q),'_',num2str(r),'_',num2str(lambda1)];

                mu_X = zeros(p,1);   mu_Y = zeros(q,1);

                %% generate data matrix
                % Identity
                if test_type == 1
                    sigma_X = eye(p);   sigma_Y = eye(q);
                else
                    if test_type == 2
                        % toeplitz matrix
                        a = 0.3;
                        c1 = a.^((1:p)-1); c2 = a.^((1:q)-1);
                        sigma_X = toeplitz(c1);   sigma_Y = toeplitz(c2);
                    else
                        % sparse inverse
                        sigma_X = zeros(p); sigma_Y = zeros(q);
                        c1 =zeros(p,1); c1(1:3,:) = [1; 0.5; 0.4]; omegaX = toeplitz(c1);    sigma_X0 = inv(omegaX);
                        c2 =zeros(q,1); c2(1:3,:) = [1; 0.5; 0.4]; omegaY = toeplitz(c2);    sigma_Y0 = inv(omegaY);
                        for i=1:p
                            for j=1:p; sigma_X(i,j)= sigma_X0(i,j)/(sqrt(sigma_X0(i,i)*sigma_X0(j,j))); end
                        end
                        for i=1:q
                            for j=1:q; sigma_Y(i,j)= sigma_Y0(i,j)/(sqrt(sigma_Y0(i,i)*sigma_Y0(j,j))); end
                        end
                        sigma_X(abs(sigma_X)<1e-3)=0;  sigma_Y(abs(sigma_Y)<1e-3)=0;
                    end
                end

                if test_type == 4
                    sigma_X = eye(p);   sigma_Y = eye(q);
                    sigma_X(ss,ss) = 0.8; sigma_Y(ss,ss) = 0.8;
                    sigma_X = sigma_X - diag(diag(sigma_X)) + eye(p);
                    sigma_Y = sigma_Y - diag(diag(sigma_Y)) + eye(q);
                end

                if test_type == 5
                    sigma_X = eye(p);   sigma_Y = eye(q);
                    sigma_X(ss,ss) = 0.5; sigma_Y(ss,ss) = 0.5;
                    sigma_X = sigma_X - diag(diag(sigma_X)) + eye(p);
                    sigma_Y = sigma_Y - diag(diag(sigma_Y)) + eye(q);
                end

                if test_type == 6
                    sigma_X = eye(p);   sigma_Y = eye(q);
                    sigma_X(ss,ss) = 0.3; sigma_Y(ss,ss) = 0.3;
                    sigma_X = sigma_X - diag(diag(sigma_X)) + eye(p);
                    sigma_Y = sigma_Y - diag(diag(sigma_Y)) + eye(q);
                end
                % vector and rho
                u1= zeros(p,r);  v1 = zeros(q,r);


                lambda = diag( [0.9;0.8*ones(r-1,1);] );

                u1(s,(1:r)) = randi( [-2, 2], size(u1(s,(1:r))));
                Tss = sigma_X(s, s);
                u1 = u1 / sqrtm(u1(s,1:r)' * Tss * u1(s,1:r)+ 1e-13*eye(r));

                v1(s,(1:r)) = randi( [-2, 2], size(v1(s,(1:r))));
                Tss = sigma_Y(s, s);
                v1 = v1 / sqrtm(v1(s,1:r)' * Tss * v1(s,1:r)+ 1e-13*eye(r));


                [u_real, ~, ~] = svd(u1, 0);
                [v_real, ~, ~] = svd(v1, 0);
                %generate covariance matrix
                sigma_XY = sigma_X*u1*lambda*v1'*sigma_Y;




                %Data =  mvnrnd([mu_X; mu_Y] ,[sigma_X,sigma_XY; sigma_XY', sigma_Y],10*N); % data matrix

                Data = rand(2*N,p+q);


                Xtrain = Data(1:N,1:p);     Xtest = Data(N+1:2*N,1:p);
                Ytrain = Data(1:N,p+1:p+q); Ytest = Data(N+1:2*N,p+1:p+q);




                [xhat,~,SXX,SYY] = scca_init(Xtrain, Ytrain, r,0.55, 1e-4, 10);
                [U0, ~, V0] = svds(xhat, r);
                Uinit_1 = U0(:,1:r);
                Vinit_1 = V0(:,1:r);
                [u0hat, ~,~] = svd(Uinit_1,'econ');  [v0hat,~,~] = svd(Vinit_1,'econ');
                [~,~,rho_init]  = canoncorr(Xtest * u0hat, Ytest * v0hat);
                lossu_init = norm(u0hat * u0hat'  - u_real * u_real', 'fro')^2;
                lossv_init = norm(v0hat * v0hat'  - v_real * v_real', 'fro')^2;


                %  CoLAR
                [Uhat, Vhat] = scca_refine(Xtrain,Ytrain, Uinit_1,Vinit_1, r, lambda1, lambda2);


                [X,Y,XtY,M1,M2] = normalize(Xtrain,Ytrain);

                [Up,Vp] = scca_penalty(M1,M2,XtY,X,Y,Uinit_1,Vinit_1,0.1*lambda1,0.1*lambda2);



                A = struct();
                A.applyA = @AXu;
                A.applyAT = @AtXu;

                B = struct();
                B.applyB = @BYv;
                B.applyBT = @BtYu;

                f = struct();
                f.cost_grad = @cca_cost_grad;
                f.data = {XtY};

                h = struct();
                h.cost = @(U,V,lambda1,lambda2) lambda1*norm(svd(U),1) + lambda2*norm(svd(V),1);
                h.prox = @proxNuclear;
                h.data = {lambda1,lambda2};


                if isLasso==1
                    A = struct();
                    A.applyA = @(X) X;
                    A.applyAT = @(X) X;

                    B = struct();
                    B.applyB = @(X) X;
                    B.applyBT = @(X) X;
                    h = struct();
                    h.cost = @(U,V,lambda1,lambda2) lambda1*(sum(vecnorm(U,2,2)) + sum(vecnorm(V,2,2)));
                    h.prox = @proximal_l211;
                    h.data = {lambda1, lambda2};

                end

                manifold = productmanifold(struct('U', stiefelgeneralizedfactory(p,r,M1),......
                    'V', stiefelgeneralizedfactory(q,r,M2)));

                % manifold = productmanifold(struct('U', stiefelfactory(p,r),......
                %     'V', stiefelfactory(q,r)));
                %Uinit_1 = Uhat;Vinit_1 = Vhat;
                UV.U = Uinit_1/sqrtm(Uinit_1'*M1*Uinit_1);
                UV.V = Vinit_1/sqrtm(Vinit_1'*M2*Vinit_1);



                manpg_para = lambda1;
                Uinit_manpg = Uinit_1;
                Vinit_manpg = Vinit_1;
                option_amanpg.b1 = manpg_para;%*sqrt((r+log(p))/N);
                option_amanpg.b2 = manpg_para;%*sqrt((r+log(q))/N);
                option_amanpg.maxiter = 5e2;    option_amanpg.tol =5e-7*q*r;   option_amanpg.inner_tol = 1e-10;
                option_amanpg.n = r;  % column number
                option_amanpg.q = q;
                option_amanpg.p = p;
                option_amanpg.X0 = Uinit_manpg;  %initial point
                option_amanpg.Y0 = Vinit_manpg;  % initial point

                option_amanpg.Xtest = Xtest;  option_amanpg.Ytest = Ytest;

                %[xhat,~] = scca_init(Xtrain, Ytrain, r,option.tau1, 1e-4, 1);
                fprintf('=========================================================\n');

                fprintf('running A-Manpg............\n');
                %% amanpg_init_1

                [result_manpg_alt] = scca_manpg(X,Y,XtY,M1,M2,option_amanpg,'l21');
                F_ManPG = result_manpg_alt.Fval;

                [result_amanpg_alt] = scca_amanpg(X,Y,XtY,M1,M2,option_amanpg,'l21');
                



                options_mialm.stepsize = 1/(2*abs(svds(full(XtY),1)));
                options_mialm.max_iter = 100;     options_mialm.maxitersub = 100;
                options_mialm.tau = 0.8;          options_mialm.rho = 1.5;
                options_mialm.nu0 = svds(X,1)^1*1 ; options_mialm.tol = 1e-7*q*r;
                options_mialm.gtol0 = 1;          options_mialm.gtol_decrease = 0.8;
                options_mialm.X0 = UV;      options_mialm.verbosity = 0;
                options_mialm.verbosity = 1;

                options_mialm.flag = 1;  options_mialm.opt = F_ManPG;
                [X_mialm1,Z1_mialm1,Z2_mialm1,out_mialm1] = mialm(A,B, manifold, f, h, options_mialm);

                options_mialm.flag = 2;
                [X_mialm2,Z1_mialm2,Z2_mialm2,out_mialm2] = mialm(A,B, manifold, f, h, options_mialm);



                f = struct();
                f.cost_grad = @partial_cca_cost_grad;
                f.data = {X,Y};

                options_ial3 = options_mialm;
                options_ial3.maxitersub = 10;
                options_ial3.rho = 2;
                options_ial3.batchsize = N/1000;
                options_ial3.max_iter = 100;
                %options_ial3.nu0 = 10;
                options_ial3.m = N; options_ial3.opt = F_ManPG;
                [X_stomialm,Z1_stomialm,Z2_stomialm,out_stomialm] = stomialm(A, B, manifold, f, h, options_ial3);

               

                minobj  = min([F_ManPG,result_amanpg_alt.Fval,out_mialm1.obj,out_mialm2.obj]);

                out_stomialm.obj_arr = out_stomialm.obj_arr - minobj;
                out_mialm1.obj_arr = out_mialm1.obj_arr - minobj;
                out_mialm2.obj_arr = out_mialm2.obj_arr - minobj;
                
                result_manpg_alt.arrF = result_manpg_alt.arrF - minobj;
                result_amanpg_alt.arrF = result_amanpg_alt.arrF - minobj;


                
                f0 = figure;

                plot(out_mialm1.time_arr(1:1:out_mialm1.iter+1),log(out_mialm1.obj_arr(1:out_mialm1.iter+1)),'-.*','LineWidth',1);
                hold on;
                plot(out_mialm2.time_arr(1:1:out_mialm2.iter+1),log(out_mialm2.obj_arr(1:out_mialm2.iter+1)),'-o','LineWidth',1);
                plot(out_stomialm.time_arr(1:1:out_stomialm.iter+1),log(out_stomialm.obj_arr(1:out_stomialm.iter+1)),'->','LineWidth',1);
                plot(result_manpg_alt.arrtime(1:1:result_manpg_alt.iter-1),log(result_manpg_alt.arrF(1:result_manpg_alt.iter-1)),'x-','LineWidth',1);
                plot(result_amanpg_alt.arrtime(1:1:result_amanpg_alt.iter-1),log(result_amanpg_alt.arrF(1:result_amanpg_alt.iter-1)),'*-','LineWidth',1);
                %plot(sub_time(1:5:maxit_att_Rsub),log(sub_obj(1:5:maxit_att_Rsub)),'-o','LineWidth',1);

                legend('ManIAL-I','ManIAL-II','stoManIAL','ManPG','AManPG');
                %legend('ManIAL-I','ManIAL-II','AManPG');
                xlabel('CPU time','FontName','Arial');
                ylabel('$F(x^k)$','interpreter','latex','FontName','Arial');
                name = sprintf('fig//compare_obj-sub-%d-%d-%.2f.png', p,r,lambda1);
                saveas(gcf, name,'png');
                 close(f0);





            end

            % save_path = strcat(save_root_res1,basename,'.mat');
            %save(save_path, 'Trace', 'Lasso', 'Init', 'Pena','p','q','r','u_trace','v_trace','u_lasso','v_lasso','u_pena','v_pena','u_real','v_real' );
        end
    end
end


disp(newline);
disp(table_str);
save_path = strcat(save_root,mat2str(test_type),'random.txt');
fid = fopen(save_path,'w+');
fprintf(fid,'%s',table_str);

    function AX = AXu(U)
        AX = zeros(N,p*r);
        for k=1:r
            AX(:,(k-1)*p+1:k*p) = X.*repmat(U(:,k)',N,1);
        end
    end


    function BY = BYv(V)
        BY = zeros(N,q*r);
        for k=1:r
            BY(:,(k-1)*q+1:k*q) = Y.*repmat(V(:,k)',N,1);
        end
    end

    function AtX = AtXu(W)
        AtX = zeros(p,r);
        for k=1:r
            AtX(:,k) = sum(X.*W(:,(k-1)*p+1:k*p))';
        end
    end


    function BtY = BtYu(W)
        BtY = zeros(q,r);
        for k=1:r
            BtY(:,k) = sum(Y.*W(:,(k-1)*q+1:k*q))';
        end
    end



    function [f,g] = cca_cost_grad(UV,XtY)
        g.U = -XtY*UV.V;  g.V = -XtY'*UV.U;
        f = sum(sum((g.V).*(UV.V)));
    end



    function g = proxNuclear(U,V,mu,lambda1,lambda2)
        [g.U,~] = prox_nuclear(U,lambda1*mu);
        [g.V,~] = prox_nuclear(V,lambda2*mu);
    end

    function g = proxLasso(U,V,mu,lambda1,lambda2)
        g.U = max(abs(U) - mu*lambda1,0).* sign(U);

        g.V = max(abs(V) - mu*lambda2,0).* sign(V);
    end


    function g = proximal_l211(  U,V,mu,lambda1,lambda2 )
        [ g.U, ~,~] = proximal_l21(  U ,mu*lambda1,1 );
        [ g.V, ~,~] = proximal_l21(  V ,mu*lambda2,1 );
    end



    function [f,g] = partial_cca_cost_grad(UV,X,Y,sample)
        del = length(sample)/N;
        XtY = X(sample,:)'*Y(sample,:);
        g.U = -XtY*UV.V*del;  g.V = -XtY'*UV.U*del;
        f = sum(sum((g.V).*(UV.V)))*del;
    end




end
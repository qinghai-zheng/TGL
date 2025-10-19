function [W,S] = main_TGL(X, lambda_1, lambda_2)
% MAIN_TGL: the main function of our Tensorized Grap Learning
% Accepted by IEEE TKDE
% zhengqinghai@fzu.edu.cn

[~, num_sam] = size(X);

W_1 = zeros(num_sam, num_sam);
W_2 = zeros(num_sam, num_sam);
P = eye(num_sam);

M = zeros(num_sam, num_sam); 

S = cell(1,2);
W = cell(1,2);
Y = cell(1,2);

for k=1:2
    S{k} = zeros(num_sam, num_sam);
    W{k} = zeros(num_sam, num_sam);
    Y{k} = zeros(num_sam, num_sam); 
end

sT = [num_sam, num_sam, 2];

Isconverg = 0;
max_iter = 100;
epson = 1e-7;
mu = 10e-5; max_mu = 10e10; pho_mu = 2;
iter = 0;

while(Isconverg == 0)
    dist_X = L2_distance(X,X);
    for i = 1:num_sam
        W_1_tmp = S{1}(:,i) - (Y{1}(:,i) + lambda_1*dist_X(:,i))/mu;
        W_1(:,i) = EProjSimplex(W_1_tmp);
    end
    W{1} = W_1;
    
    for i = 1:num_sam
        W_2_tmp = (S{2}(:,i)+P(:,i)- Y{2}(:,i)/mu - M(:,i)/mu)/2;
        W_2(:,i) = EProjSimplex(W_2_tmp);
    end
    W{2} = W_2;
    
    P_tmp_l = 2*lambda_2*(X'*X) + mu*eye(num_sam);
    P_tmp_r = 2*lambda_2*(X'*X) + M + mu*W_2;
    P = P_tmp_l\P_tmp_r;

    W_tensor = cat(3, W{:,:});
    Y_tensor = cat(3, Y{:,:});
    w = W_tensor(:);
    y = Y_tensor(:);
    [s, ~] = wshrinkObj(w + 1/mu*y,1/mu,sT,0,3);
    S_tensor = reshape(s, sT);
    S{1} = S_tensor(:,:,1);
    S{2} = S_tensor(:,:,2);

    y = y + mu*(w - s);
    Y_tensor = reshape(y, sT);
    Y{1} = Y_tensor(:,:,1);
    Y{2} = Y_tensor(:,:,2);
    M = M + mu*(W{2}-P);
    mu = min(mu*pho_mu, max_mu);

    Isconverg = 1;
    for k = 1:2
        if norm(W{k}-S{k},inf)>epson
            Isconverg = 0;
        end
    end
    if norm(W{2}-P,inf)>epson
        Isconverg = 0;
    end
    
    iter = iter + 1;
    if (iter > max_iter)
        Isconverg  = 1;
    end
end

end


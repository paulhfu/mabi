function [u] = tv_Minimization(v, w, bet, K, maxiter)
% Algorithm based on: Chambolle, A.: An algorithm for total variation minimization and applications. J. Math. Imaging Vis.20,89–97 (2004)
% The objective this algorith is applied on is different though. It is the
% first step(line 11) on Algorithm 1 in: Cauchy Noise Removal by Nonconvex ADMM with Convergence Guarantees,
% Jin-Jin Mei et al, Published online: 30 May 2017 
b = bet; %lambda
lbd = 1/b;
g = v - w / b;
p = ones([size(v) 2]);

Nsig = normX(g-mean(mean(g)));
tau = 1/5;

for j=1:maxiter
    for i=1:10
        div_p = divergence(p(:,:,1), p(:,:,2));
        %[grad_x, grad_y] = gradient(div_p - g * b);
        [grad_x, grad_y] = gradient(div_p - g / lbd);
        grad = grad_x;
        grad(:,:,2) = grad_y;

        nom = p + tau .* grad;
        %dnom = 1 + tau .* normXX(grad);
        dnom = 1 + tau * abs(grad); %component-wise absolute value
        p = nom ./ dnom;
    end
    div_p = divergence(p(:,:,1), p(:,:,2));
    lbd = Nsig / normX(div_p); %lambda
    g = v - w / b;
    Nsig = normX(g-mean(mean(g)));
    tau = tau / 2;
    %u = g - div_p/b;
    u = g - lbd*div_p; %(7) and Thm. 3.1
end


function [u] = tv_MinimizationChambolle(v, w, bet, K, maxiter)
% Algorithm based on: Chambolle, A.: An algorithm for total variation minimization and applications. J. Math. Imaging Vis.20,89�97 (2004)
% The objective this algorith is applied on is different though. It is the
% first step(line 11) on Algorithm 1 in: Cauchy Noise Removal by Nonconvex ADMM with Convergence Guarantees,
% Jin-Jin Mei et al, Published online: 30 May 2017 
b = bet;
lbd = 1/b;
g = v - w / b;
p = zeros([size(v) 2]);

Nsig = normX(g-mean(mean(g)));
tau = 1/10;

for j=1:maxiter
    for i=1:5
        div_p = divergence(p(:,:,1), p(:,:,2));
        [grad_x, grad_y] = gradient(div_p - g / lbd);
        grad = grad_x;
        grad(:,:,2) = grad_y;

        nom = p + tau .* grad;
        dnom = 1 + tau * abs(grad); %component-wise absolute value
        p = nom ./ dnom;
    end
    div_p = divergence(p(:,:,1), p(:,:,2));
    lbd = Nsig / normX(div_p);
    b = 1 / lbd;
    g = v - w / b;
    Nsig = normX(g-mean(mean(g)));
    u = g - lbd*div_p; %(7) and Thm. 3.1
end


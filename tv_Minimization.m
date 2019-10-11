function [u] = tv_Minimization(v, w, bet, K, maxiter)
% Algorithm based on: Chambolle, A.: An algorithm for total variation minimization and applications. J. Math. Imaging Vis.20,89–97 (2004)
% The objective this algorith is applied on is different though. It is the
% first step(line 11) on Algorithm 1 in: Cauchy Noise Removal by Nonconvex ADMM with Convergence Guarantees,
% Jin-Jin Mei et al, Published online: 30 May 2017 
b = bet;
g = v - w ./ b;
p = ones([size(v) 2]);
Nsig = normX(g-mean(mean(g)));
tau = 1/5;

for j=1:maxiter
    for i=1:10
        div_p = divergence(p(:,:,1), p(:,:,2));
        [grad_x, grad_y] = gradient(div_p - g .* b);
        grad = grad_x;
        grad(:,:,2) = grad_y;

        nom = p + tau .* grad;
        dnom = 1 + tau .* normXX(grad);
        p = nom ./ dnom;
    end
    div_p = divergence(p(:,:,1), p(:,:,2));
    b = Nsig / (normX(div_p) / b);
    g = (v - w) ./ b;
    Nsig = normX(g-mean(mean(g)));
    tau = tau / 2;
    u = g - div_p/b;
end


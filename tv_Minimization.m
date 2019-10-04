function [u] = tv_Minimization(u_old, v, w, b, K)
% Algorithm based on: Chambolle, A.: An algorithm for total variation minimization and applications. J. Math. Imaging Vis.20,89–97 (2004)
% The objective this algorith is applied on is different though. It is the
% first step(line 11) on Algorithm 1 in: Cauchy Noise Removal by Nonconvex ADMM with Convergence Guarantees,
% Jin-Jin Mei et al, Published online: 30 May 2017 
g = v - w ./ b;
p = ones([size(u_old) 2]);
Nsig = normX(g-mean(mean(g))) + eps;
tau = 1/8;
for i=1:100
    nom = p + tau .* (gradient(divergence(p(:,:,1), p(:,:,2))-g .* b));
    dnom = 1 + tau .* (gradient(divergence(p(:,:,1), p(:,:,2))-g .* b));
    p = nom ./ dnom;
    b = (Nsig / normX(divergence(p(:,:,1), p(:,:,2)))) / b;
    g = (v - w) ./ b;
    Nsig = normX(g-mean(mean(g)));
    tau = tau / 2;
end
u = g - divergence(p(:,:,1), p(:,:,2));
end


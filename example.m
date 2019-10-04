                  clc; close all; clear all

% init random seed to reproduce randomness
% rng(42)

% call with some parameters
img = randi(500);
sigma = 0.1;
[b, u0] = denoisingLoadData('BSDS500', img, true, 'Cauchy', sigma);


h = size(u0, 1); % image height
w = size(u0, 2); % image width
c = size(u0, 3); % number of channels

% figure(1)
% subplot(1,2,1); imshow(u0); title('Original');
% subplot(1,2,2); imshow(reshape(b,[h w c])); title('Degraded');

% vecim = u0(:);
% sz = size(u0);
% img = reshape(vecim, sz);
b = reshape(b,[h w c]);
quality = 0.1;
u_all = b;
lbd = 0.01;
bet = 0.01;
K = fspecial('gaussian', [9 9], 4);

for c=1:3
    f = squeeze(b(:,:,c));
    u = squeeze(b(:,:,c));
    v = zeros(size(f));
    w = zeros(size(f));
    while normX(f-u)<quality
        u = tv_Minimization(u, v, w, bet, K);
        
        fnc = @(v_n)( lbd/2 .* (1/(y^2 + dotX((v_n-f), (v_n-f))) .* ones(size(f)))...
        + bet * (u - v_n + w/bet) );
        fncdot = @(v_n)( (-lbd .* (v_n-f)) ./ (y^2 + dotX((v_n-f), (v_n-f))) - bet .* v_n );
        
        v = newton(fnc, fncdot, v, 1000, 0.001);
        w = w + bet .* (u - v);
    end
    u_all(:,:,c) = u;
end

function ret = PSNR(u_true, u_pred)
    ret = 20 * log10(255 / normX(u_true-u_pred));
end
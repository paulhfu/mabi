clc; close all; clear all

% init random seed to reproduce randomness
% rng(42)

% call with some parameters
img = randi(500);
sigma = 0.01;
[b, u0] = denoisingLoadData('BSDS500', img, true, 'Cauchy', sigma);


h = size(u0, 1); % image height
w = size(u0, 2); % image width
c = size(u0, 3); % number of channels

figure(1)
subplot(1,2,1); imshow(u0); title('Original');
subplot(1,2,2); imshow(reshape(b,[h w c])); title('Degraded');

b = reshape(b,[h w c]);
d = min([h w]);
u0 = u0(1:d, 1:d, :);
b = b(1:d, 1:d, :);

G = gradient_discrete_4('d', d);
quality = 25;
u_all = b;
lbd = 1.5;
y = 10;
bet = round(0.005 + lbd / (y^2), 2); % ceil to nearest second decimal
K = fspecial('gaussian', [9 9], 4);
log1 = [];
log2 = [];
for c=1:3
    f = squeeze(b(:,:,c));
    u = f;
    w = ones(size(f));

    fnc = @(v_n) (lbd * (2*(v_n-f) ./ (y^2 + (v_n-f).^2)) - bet * (u - v_n + w/bet));
    fncdot = @(v_n) (lbd * (y^2-(v_n-f).^2) ./ (y^2 + (v_n-f).^2).^2 + bet * ones(size(v_n)));
    
    v = newton(fnc, fncdot, u, 0.1, 30);
    w = w + bet .* (u - v);
    
    log2 = [log2 PSNR(u0(:,:,c), u)];
    while log2(end)<quality
        g = v - w / bet;
        g = reshape(g,[size(g,1)*size(g,2) 1]);
        u = reshape(u,[size(u,1)*size(u,2) 1]);
        l = 1/bet;
        if log2(end) > 18
            maxiter_d = 300;
            maxiter_n = 50;
            lbd = 1/4;
        else
            maxiter_d = 50;
            maxiter_n = 10;
        end
        u = denoising_anisotrop_tv(u, 'lambda', l, 'v', g, 'x0', u, 'maxIter', maxiter_d);
        u = reshape(u,[d d]);
        
        fnc = @(v_n) (lbd * ((v_n-f) ./ (y^2 + (v_n-f).^2)) - bet * (u - v_n + w/bet));
        fncdot = @(v_n) (lbd * (y^2-(v_n-f).^2) ./ (y^2 + (v_n-f).^2).^2 + bet * ones(size(v_n)));
        
        v = newton(fnc, fncdot, v, 0.1, maxiter_n);
        w = w + bet .* (u - v);
        
        
        log1 = [log1 normX(u-u0(:,:,c))];  % cannot preallocate here since number of iterations is unclear
        log2 = [log2 PSNR(u0(:,:,c), u)];
        
        figure(2)
        subplot(2,1,1)
%         plot(1:length(log1),log1, '-b')
        plot(1:length(log2),log2, '-r')
        hold on
        plot(1:length(log2),log2, '-r')
%         legend('norm difference', 'PSNR')
        subplot(2,1,2)
        imshow(u)
    end
    u_all(:,:,c) = u;
end

function ret = PSNR(u_true, u_pred)
    ret = 20 * log10(255 / normX(u_true-u_pred));
end
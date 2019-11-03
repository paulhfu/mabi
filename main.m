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

b = reshape(b,[h w c]);  % reshape two two dim
% crop to largest square in image, init variables
d = min([h w]);
u0 = u0(1:d, 1:d, :);
b = b(1:d, 1:d, :);
u_all = zeros(size(b));

% init weight parameters
lbd = 0.1;
y = sigma;
bet = round(0.005 + lbd / (y^2), 2)*5; % ceil to nearest second decimal

% set max iterations for newton method and primal dual algorithm
maxiter_d = 10;
maxiter_n = 5;

for c=1:3
    log1 = [inf]; %log of norm difference
    log2 = [-inf]; %log of PSNR
    f = b(:,:,c);
    u = f; %init with dimensions of observation
    w = ones(size(f));

    v = u; %aux. var. (10)
    
    log1 = [log1 normX(u-u0(:,:,c))];
    log2 = [log2 PSNR(u0(:,:,c), u)];   
    
    while log2(end)>log2(end-1) 
        g = v - w / bet;
        g = reshape(g,[size(g,1)*size(g,2) 1]);
        u = reshape(u,[size(u,1)*size(u,2) 1]);
        l = 1/bet;

        u = denoising_isotrop_tv(u, 'lambda', l, 'v', g, 'maxIter', maxiter_d, 'l', 0 ,'u', 1);
        u = reshape(u,[d d]);
        
        % first derivative for objective function of v
        fnc = @(v_n) (lbd * ((v_n-f) ./ (y^2 + (v_n-f).^2)) - bet * (u - v_n + w/bet));
        % second derivative for objective function of v
        fncdot = @(v_n) (lbd * (y^2-(v_n-f).^2) ./ (y^2 + (v_n-f).^2).^2 + bet * ones(size(v_n)));
        
        v = newton(fnc, fncdot, v, 0.0001, maxiter_n);
        w = w + bet .* (u - v);
        
        % logging the performance measures
        log1 = [log1 ssim(u, u0(:,:,c))];
        log2 = [log2 PSNR(u0(:,:,c), u)];
        
        % plot current state
        figure(2)
        subplot(2,2,1)
        plot(1:length(log2),log2, '-r')
        ylabel('PSNR')
        subplot(2,2,2)
        plot(1:length(log1),log(log1),'-b')
        ylabel('SSIM')
        hold on
        subplot(2,2,3)
        imshow(u)
    end
    u_all(:,:,c) = u;
end
figure(3)
subplot(1,3,1); imshow(u0); title('Original');
subplot(1,3,2); imshow(u_all); title('Denoised');
subplot(1,3,3); imshow(b); title('Noisy');

function ret = PSNR(u_true, u_pred)
% peak signal to noise ratio performance measure
    ret = 20 * log10(255 / normX(u_true-u_pred));
end
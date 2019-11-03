clc; close all; clear all

% get random image and a corresponding noisy image from BSDS500
% img = randi(500);
img = 20;
sigma = 0.1;
[b, u0] = denoisingLoadData('BSDS500', img, true, 'Cauchy', sigma);


h = size(u0, 1); % image height
w = size(u0, 2); % image width
c = size(u0, 3); % number of channels

b = reshape(b,[h w c]);  % reshape to image dimensions
% crop to largest square in image
d = min([h w]);  % side length of largest square in image
u0 = u0(1:d, 1:d, :);  % cropped original image
b = b(1:d, 1:d, :);  % cropped noisy image
u_all = zeros(size(b));  % container for denoised image
quality = 0.01;

% init weight parameters
lbd = 0.2;
y = sigma;
bet = round(0.005 + lbd / (y^2), 2)*10; % ceil to nearest second decimal

% set max iterations for newton method and primal dual algorithm
maxiter_d = 10;
maxiter_n = 5;

for c=1:3
    % init performance measures
    log1 = [inf];
    log2 = [-inf];
    f = b(:,:,c);
    u = f; %init with dimensions of observation
    w = ones(size(f));

    v = u; %aux. var. (10)
    
    % logging performance measures
    log1 = [log1 ssim(u, u0(:,:,c))];  % structural similarity measure (close to one is optimal)
    log2 = [log2 PSNR(u0(:,:,c), u, d)];  % peak signal to noise ratio (higher value correponds to better performance) 
    itr = 1;
    while log2(end)-log2(end-1) > quality  % iterate as long as performance is improving
        itr = itr + 1;
        g = v - w / bet;  % prior for denoising_isotrop_tv
        % vectorize
        g = reshape(g,[size(g,1)*size(g,2) 1]);
        u = reshape(u,[size(u,1)*size(u,2) 1]);
        lambda = 1/bet;

        % update primal solution
        u = denoising_isotrop_tv(u, 'lambda', lambda, 'v', g, 'maxIter', maxiter_d, 'l', 0 ,'u', 1);
        u = reshape(u,[d d]);  % back to image dimensions
        
        % first derivative for objective function of v
        fnc = @(v_n) (lbd * ((v_n-f) ./ (y^2 + (v_n-f).^2)) - bet * (u - v_n + w/bet));
        % second derivative for objective function of v
        fncdot = @(v_n) (lbd * (y^2-(v_n-f).^2) ./ (y^2 + (v_n-f).^2).^2 + bet * ones(size(v_n)));
        
        % update auxiliary variable w´v
        v = newton(fnc, fncdot, v, maxiter_n);
        % update dual variable w
        w = w + bet .* (u - v);
        
        % logging performance measures
        log1 = [log1 ssim(u, u0(:,:,c))];
        log2 = [log2 PSNR(u0(:,:,c), u, d)];
        
        % plot current state
        figure(c+1)
        suptitle({'','',sprintf('beta=%-5.2f; lambda=%-5.2f; gamma=%-5.2f;\n channel:%i; iterations:%i maxPsnr:%-5.2f', bet, lbd, y, c, itr, log2(end)),' ',' ',''})
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
figure(5)
suptitle('Lineup')
subplot(1,3,1); imshow(u0); title('Original');
subplot(1,3,2); imshow(u_all); title('Denoised');
subplot(1,3,3); imshow(b); title('Noisy');

function ret = PSNR(u_true, u_pred, n)
% peak signal to noise ratio performance measure of 1-normalized images 
    ret = 20 * log10(n / normX(u_true-u_pred));
end
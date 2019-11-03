clc; close all; clear all
%this is for testing the FGPA algorithm without ADMM

% get random image and a corresponding noisy image from BSDS500
% img = randi(500);
img = 20;
sigma = 0.1;
[b, u0] = denoisingLoadData('BSDS500', img, true, 'Cauchy', sigma);

h = size(u0, 1); % image height
w = size(u0, 2); % image width
c = size(u0, 3); % number of channels

b = reshape(b,[h w c]);
d = min([h w]);
u0 = u0(1:d, 1:d, :);
b = b(1:d, 1:d, :);
u_all = b;

G = gradient_discrete_4('d', d);
psnr = 0;
for c=1:3
    g = b(:,:,c);
    g = reshape(g,[size(g,1)*size(g,2) 1]);
    u = g;
    u = denoising_anisotrop_tv(u, 'lambda', 0.5, 'v', g, 'maxIter', 200, 'l', 0 ,'u', 1);
    u = reshape(u,[d d]);

    u_all(:,:,c) = u;
    fprintf('PSNR: %d; SSIM: %d \n', PSNR(u0(:,:,c), u, d), ssim(u0(:,:,c), u));
    psnr = psnr + PSNR(u0(:,:,c), u, d);
end
psnr = psnr/3;
figure(1)
suptitle(sprintf('Lineup\niterations:200 maxPsnr:%-5.2f', psnr))
subplot(1,3,1); imshow(u0); title('Original');
subplot(1,3,2); imshow(u_all); title('Denoised');
subplot(1,3,3); imshow(b); title('Noisy');


function ret = PSNR(u_true, u_pred, n)
    ret = 20 * log10(n / normX(u_true-u_pred));
end
% Type 'help denoisingLoadData'
clc; close all; clear all
%this is for testing the FGPA algorithm without ADMM

% init random seed to reproduce randomness
% rng(42)

% call with some parameters
img = randi(500);
sigma = 0.1;
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
u_all = b;

G = gradient_discrete_4('d', d);

for c=1:3
    g = b(:,:,c);
    g = reshape(g,[size(g,1)*size(g,2) 1]);
    u = g;
    u = denoising_anisotrop_tv(u, 'lambda', 0.5, 'v', g, 'maxIter', 200, 'l', 0 ,'u', 1);
    u = reshape(u,[d d]);

    u_all(:,:,c) = u;
    fprintf('PSNR: %d; Norm: %d \n', PSNR(u0(:,:,c), u), normX(u-u0(:,:,c)));
end
figure(2)
subplot(1,2,1); imshow(u_all); title('denoised');
subplot(1,2,2); imshow(b); title('Degraded');


function ret = PSNR(u_true, u_pred)
    ret = 20 * log10(255 / normX(u_true-u_pred));
end
% Type 'help denoisingLoadData'
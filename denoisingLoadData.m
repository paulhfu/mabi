function [b, u_orig] = denoisingLoadData( dataset, instance, color, noiseType, sigma )
%[b, u_orig] = denoisingLoadData( dataset, instance, color, noiseType, sigma )
%
%   INPUT:
%   -------------------------------------------------------------
%   dataset           {'BSDS500'}
%   instance          the instance number in the dataset
%   color             true | false = grayscale
%   noiseType         {'Gaussian'}, {'Impulse'}, {'Cauchy'}
%   sigma             deviation/ratio of noise
%
%   OUTPUT:
%   --------------------------------------------------------------
%   b                 observed data     NxC
%   u_orig            original image    WxHxC

    % Default value for sigma
    if nargin < 5
        sigma = 0.05;
    end
    
    % Default value for noiseType
    if nargin < 4
        noiseType = 'Gaussian';
    end
    
    % Default value for color
    if nargin < 3
        color = false;
    end    
    
    % Default value for instance
    if nargin < 2
        instance = 1;
    end
    
    % Default value for dataset
    if nargin < 1
        dataset = 'BSDS500';
    end
    
    % Add cauchy toolbox to path
    S = dbstack('-completenames');
    currentFolder = fileparts( S(1).file );
    addpath( [currentFolder, '/cauchy'] )

    %% Load Data
    mfn  = mfilename;
    mffn = mfilename('fullpath');
    datapath = [mffn(1:end-numel(mfn)),'/data/',dataset,'/'];
    files    = dir([datapath,'*.jpg*']);
    if size(files,1) < 1
        error(['Error: no image data found for dataset ' dataset])
    end
    if instance < 1 || instance > size(files,1)
        error(['Error: instance must be in the range of [1,' num2str(size(files,1)) '] for dataset ' dataset])
    end
    
    u_orig   = im2double(imread([datapath,files(instance).name]));
    
    % convert to grayscale
    if color == false
        u_orig = rgb2gray(u_orig);
    end  
    
    height = size(u_orig, 1);
    width = size(u_orig, 2);
    pixels = height*width;    
    channels = size(u_orig,3);
    
    %% Add noise
    switch noiseType
      case 'Gaussian'
        b = imnoise(u_orig,'gaussian',0,sigma^2);
      case 'Impulse'
        b = imnoise(u_orig,'salt & pepper',sigma);
      case 'Cauchy'
        noise = cauchyrnd(0, sigma, height, width, channels); % zero mean cauchy noise
        b = u_orig + noise;
      otherwise
        error('Error: unsupported type of noise')
    end
    
    b = reshape(b, pixels, channels );
    
end
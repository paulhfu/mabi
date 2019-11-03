function [x,iter] = denoising_isotrop_tv(varargin)
%
% argmin_{l<=x<=u} lbd*|G*x|_1 + 0.5*|x-v|_2^2
%

ip = inputParser;

addRequired(ip,'v',@(x) size(x,2) == 1 && round(sqrt(length(x))) == sqrt(length(x)) );
addParameter(ip,'maxIter',100,@(x) x > 1);
addParameter(ip,'lambda',1e-3,@(x) x > 0);
addParameter(ip,'tol',1e-6,@(x) x > 0);
addParameter(ip,'G',[],@(x) ~isempty(x));
addParameter(ip,'l',-inf);
addParameter(ip,'u',inf);
addParameter(ip,'disp',inf,@(x) x > 0);
addParameter(ip,'out',0,@(x) x > 0);
addParameter(ip,'x0',[],@(x) ~isempty(x));

parse(ip,varargin{:});
par = ip.Results;

lbd = par.lambda;
v = par.v;
n = length(v);
d = sqrt(n);
l = par.l; u = par.u;

if( isempty(par.G) )
  G = gradient_discrete_4('d',d);
else
  G = par.G;
end


maxiter = par.maxIter;

t0 = 1;
p0 = zeros(size(G,1),1);

if( ~isempty(par.x0) )
  x0 = par.x0;
else
  x0 = zeros(n,1);
end
r = (1/(lbd*8))*G*x0;

if( par.disp < inf ), par.out = 1; end

if( par.out > 0 )
  iter = zeros(maxiter,3);
end

t_time = tic;
for i=1:maxiter
   
    % Create Primal Solution
    y = v - lbd*G'*r;    % Gradient step
    x = max(l,min(u,y)); % Proximal map
    
    if( par.out > 0 )
      iter(i,1) = 0.5*norm(x-v,2)^2 + lbd*norm(G*x,1);
      iter(i,2) = norm(x0-x,2);
      iter(i,3) = toc(t_time);
    end
    
    if( mod(i,par.disp) == 0 || (i==1 && par.disp < inf) )
      fprintf('TV denoising -> %05d: obj: %10.5f  dist: %10.5f \n',i,iter(i,1),iter(i,2));
    end
    
    if( i > 1 &&  norm(x0-x,inf) < par.tol )
        if( par.disp < inf )
          fprintf('---------------------------------------------------------\n');
          fprintf('TV denoising -> %05d: obj: %10.5f  dist: %10.5f (final)\n',i,iter(i,1),iter(i,2));
          fprintf('---------------------------------------------------------\n');
        end
        break;
    end
    x0 = x;
    
    % Update Dual Variables according to isotropic TV
    p1 = r + (1/(lbd*8))*G*x;
    sz = size(p1);
    % reshape p1 to a (d,d-1) matrix for the gradient in y direction and to
    % (d-1,d) for the gradient in x direction
    p1 = reshape(p1,[sz(1)/2, 2]);
    px = reshape(p1(:,1), [d-1, d]);
    py = reshape(p1(:,2), [d, d-1]);
    % abs val of last column/row ist constrained <=1
    px(:,end) = max(-1,min(1,px(:,end)));
    py(end,:) = max(-1,min(1,py(end,:)));
    % bring p1 back to original shape
    py = reshape(py, [sz(1)/2, 1]);
    px = reshape(px, [sz(1)/2, 1]);
    % sum of squares is constrained by <=1
    %px = min(1, px.^2);
    px = px ./ max(1, sqrt(px.^2 + py.^2));
    %py = (sqrt(1-px.^2));
    py = py ./ max(1, sqrt(px.^2 + py.^2));

    p1 = [px py];
    p1 = reshape(p1,sz);
    
    
    % Extrapolation
    t1 = (1 + sqrt(1 + 4*t0^2))/2;    
    r = p1 + ((t0-1)/t1)*(p1-p0);    
    
    t0 = t1;
    p0 = p1;
    
end

% Create Primal Solution
y = v - lbd*G'*r;
x = max(l,min(u,y));

if( norm(x0-x,inf) > par.tol )
  warning('denoising_anisotrop_tv -> tolerance not reached!');
end

if( par.out > 0 )
  iter = iter(iter(:,3) > 0,:);
else
  iter = norm(x-x0,inf);
end

end


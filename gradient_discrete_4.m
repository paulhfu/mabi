function [ G ] = gradient_discrete_4(varargin)
% d -> produces a finite difference operator for a square image d*d
% n -> produces a finite difference operator for a n dimension vector

ip = inputParser;

addParameter(ip,'d',0,@(x) x > 1 && round(x) == x );
addParameter(ip,'n',0,@(x) x > 1 && round(x) == x );

parse(ip,varargin{:});
par = ip.Results;

d = par.d; n = par.n;
assert( d == 0 || n == 0 );

if( n == 0 )
  op = speye(d)-triu(ones(d,d),1)+triu(ones(d,d),2);
  op = op(1:d-1,:);
  G = [ kron(speye(d),op) ;
        kron(op,speye(d)) ];
else
  G = eye(n-1,n)-triu(ones(n-1,n),1)+triu(ones(n-1,n),2);
end

end

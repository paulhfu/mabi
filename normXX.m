function out = normXX(arg)
% returns a norm in XxX with X=R-NxN as
    out = (arg(:,:,1).^2 + arg(:,:,2).^2).^(1/2);
end


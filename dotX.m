function ret = dotX(arg1, arg2)
% returns a dot product in R-NxN as the sum of pointwise multiplication
    ret = sum(sum(arg1 .* arg2));
end
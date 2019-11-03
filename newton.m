function x = newton(f, fdot, init, maxiter)
    % newtons method
    x = init;
    for n = 1:maxiter
        d = f(x)./fdot(x);
        x = x - d;
    end
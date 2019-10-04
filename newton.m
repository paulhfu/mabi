function x = newton(f, fdot, init, maxiter, err)
    x = init;
    for n = 1:maxiter
        d = f(x)/fdot(x);
        x = x - d;
        if abs(d) < err
            return
        end
    end
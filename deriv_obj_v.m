function ret = deriv_obj_v(lambda, y, v, f, K, u, w, beta)
    ret = lambda/2 .* (1/(y^2 + dotX((v-f), (v-f))) .* ones(size(f)))...
        + beta * (K*u - v + w/beta)
end


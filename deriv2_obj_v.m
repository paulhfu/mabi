function ret = deriv2_obj_v(lambda, y, v, f, beta)
    ret = (-lambda .* (v-f)) ./ (y^2 + dotX((v-f), (v-f))) - beta .* v;
end


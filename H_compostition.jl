gradient_vec = function(beta, X, Y)
    G = X' * (-Y + (1 ./ (1 .+ exp.(-X * beta))))
    return G
end

hessian_mat = function(beta, X)
    n, p = size(X)
    H = zeros(p, p)

    for i in 1:n
        S = 1 ./ ( 1 .+ exp.(-X[i,:]' * beta))
        tmp_H = (X[i,:] * X[i,:]') * S * (1 - S)
        H += tmp_H
    end
    return H
end

### Data load

using RData

cd("C:/Users/UOS/Desktop/H-composition/data")

simul_df = load("simulation_data.RData")

gl = simul_df["gl"]
A = simul_df["A"]
newz = simul_df["newz"]
y = simul_df["y"]
size(newz)
# repeat([1], inner = 10), fill(1 ,10)

beta = fill(1, 123)

hessian_mat(beta, newz)

# ifelse.(1 .- beta .< 0, 1., 0.)
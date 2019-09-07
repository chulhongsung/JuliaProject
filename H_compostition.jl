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

using Convex, SCS

m = 4; n = 5

A = randn(m, n); b = randn(m, 1)

x = Variable(n)

problem = minimize(sumsquares(A * x - b), [x >= 0])


#####

rho = 0.5

n, p = size(newz)

### Initial 

tmp_gamma = randn(p*4)

tmp_nu = randn(p*4)

tmp_u = tmp_nu / rho

tmp_beta = fill(0, p)

Amat = fill(1, p)

### for loop

for i in 1:10000

    d = tmp_u - tmp_nu

    grad = gradient_vec(tmp_beta, newz, y)

    hm = hessian_mat(tmp_beta, newz)

    x = Variable(p)

    problem = minimize(quadform(x, rho * (A'* A + hm)) + (tmp_beta' * hm - grad - rho*d*Amat)'*x, Amat' * x = 1)

    solve!(problem, SCSSolver())

    problem.status

    x.value # optimal value

end
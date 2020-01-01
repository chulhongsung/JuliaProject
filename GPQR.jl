using Pkg, RData

cd("C:/Users/chica/Desktop/data")

kospi200 = load("kospi200.RData")

typeof(kospi200)

keys(kospi200) ### Dict type의 data에 key를 확인

kospi_mat = kospi200["df_kospi200"][2:end,]
group_inx = kospi200["group_index"]

### Data type 변환 DF -> Matrix
kospi_mat = convert(Matrix, kospi_mat);

### log return으로  변환
log_return_mat= function(x)
    
    nrow = size(x)[1]
    ncol = size(x)[2]
    
    lag1 = x[2:end,:]
    
    log_return_mat = log.(lag1) .- log.(kospi_mat[1:(end-1),:])
    
    return log_return_mat
end

rmat = log_return_mat(kospi_mat) 
size(rmat)

using Statistics

μ̂ = mean(rmat, dims = 1)

nrow = size(rmat)[1]
ncol = size(rmat)[2]

p = ncol; n = nrow

ρ = 1; τ = 0.1

tmp_z = fill(0, 2*p)
tmp_u = fill(0, 2*p+2)
tmp_β = fill(1, p)/p
tmp_β₀ = 1e-5

tmp_β̃ = vcat(tmp_β₀, tmp_β);

using Random, Distributions, LinearAlgebra

μ₀ = 0.0001

Amat = vcat(Diagonal(ones(p)), Diagonal(ones(p)), ones(p)', μ̂)
Bmat = vcat(Diagonal(-1*ones(2*p)), zeros(2*p)', zeros(2*p)')
Cmat = vcat(zeros(2*p), 1, μ₀)

X̃ = hcat(fill(-1, n), rmat)
K = hcat(zeros(p), Diagonal(ones(p)));

using StatsBase

#countmap(group_inx)
pl = sqrt.(counts(group_inx)); iter_outer = 1e+6; iter_inner = 100; 
λ₁ = 0.02; λ₂ = 0.01;

i = 1
while i <= iter_outer
    
    j = 1 
    
    while j <= iter_inner  
        
        tmp_W = Diagonal(ifelse.(abs.((X̃ * tmp_β̃) * 4) .< 1e-8, 1e+8, abs.(1/((X̃ * tmp_β̃)*4))))

        tmp_β̃ = (-1/2) * inv(X̃' * tmp_W * X̃ + (ρ/2) * (Amat * K)' * (Amat * K)) * ((τ - (1/2)) * X̃' * ones(n) +
            ρ * (Amat*K)' * ((1/ρ)*tmp_u + Bmat*tmp_z - Cmat)) 
        j += 1
    end
    
    tmp_β = tmp_β̃[2:end]
    
    v = (Amat * tmp_β - Cmat)
    v₁ = v[1:p]
    tmp_z₁ = v₁ + tmp_u[1:p]/ρ
    
    ### Proximal operator
    tmp_z₁ = ifelse.(abs.(tmp_z₁) .<= λ₁, 0, tmp_z₁ - λ₁ * sign.(tmp_z₁))
    
    v₂ = v[(p+1):end]
    u₂ = tmp_u[(p+1):end]
    
    tmp_z₂ = v₂ + u₂/ρ 
    
    for l in 1:unique(group_inx)
        plₗ = pl[l]
        v₂ₗ = v₂[group_inx .== l]
        u₂ₗ = u₂[group_inx .== l]
        
        if(norm(tmp_z₂[group_inx .== l], 2) .> λ₂ * plₗ)
            tmp_z₂[group_inx .== l] = tmp_z₂[group_inx .== l] - λ₂ * plₗ * tmp_z₂[group_inx .== l] / norm(tmp_z₂[group_inx .== l],2)
        else 
            tmp_z₂[group_inx .== l] .= 0
        end
    end        
    
    tmp_z = vcat(tmp_z₁, tmp_z₂)
    
    tmp_u = Amat * tmp_β + Bmat * tmp_z - Cmat
    
end

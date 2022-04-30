using LinearAlgebra

function proj_simplex(x)
    n = size(x)[1];
    mu = sort(x, rev=true); # mu1 >= ... >= mun
    sum_mu = cumsum(mu, dims=1);
    j = collect(1:n);
    rho = n + 1 - argmax((mu .- (sum_mu .- 1) ./ j .> 0)[end:-1:1]);
    theta = (sum_mu[rho] .- 1) / rho;
    z = max.(x .- theta, 0);
    return z;   
end

function log_opt_return_sample(m)
    """
    return oracle
     - outputs n X m matrix of log normal mixture
    """
    # asset dimension 
    n = 10;
    # generate samples from mixture distribution
    # ------------------------------------
    # means
    mu1 = [0.4, 0.2, 0.4, 0.7, 0.9, 0.3, 0.9, 0.5, 0.5, 0.1]
    mu2 = [5.1, 0.9, 0.7, 1.6, 2.2, 4.4, 3.2, 6.2, 4.2, 0.7]

    A = randn(n, n)
    A = A*A'
    B = randn(n, n)
    B = B*B'

    C1 = Diagonal(1 ./sqrt.(diag(A)))*A*Diagonal(1 ./sqrt.(diag(A)))
    C2 = Diagonal(1 ./sqrt.(diag(B)))*B*Diagonal(1 ./sqrt.(diag(B)))

    sigmas1 = [0.01, 0.03, 0.05, 0.04, 0.05, 0.09, 0.05, 0.01, 0.03, 0.12]
    sigmas2 = [0.81, 0.31, 0.74, 0.91, 0.67, 0.71, 0.31, 0.42, 0.51, 0.41]

    sigma1 = Diagonal(sigmas1)*C1*Diagonal(sigmas1)
    s1Half = sqrt(sigma1)
    sigma2 = Diagonal(sigmas2)*C2*Diagonal(sigmas2)
    s2Half = sqrt(sigma2)

    # bernoulli trials
    p = vcat([rand(1, m) .<= 0.9 for i=1:n]...)
    q = 1 .- p

    # samples from first distribution
    r1 = exp.(hcat([mu1 for i=1:m]...)) + s1Half*randn(n, m)
    r1 = p.*r1

    # samples from second distribution
    r2 = exp.(hcat([mu2 for i=1:m]...)) + s2Half*randn(n, m)
    r2 = q.*r2

    # mixture
    R = r1 + r2
    return R
end


using Distributions
using Roots
using Statistics
import Random

struct Particle{T}
    x::T
    Particle(x::T) where {T} = new{T}(x)
end

Base.length(::Particle{T}) where {T} = fieldcount(T)

function Statistics.std(ps::Vector{Particle{T}}) where {T}
    n = fieldcount(T)
    res = Array{Float64}(undef, n)
    for i = 1:n
        res[i] = std((p.x[i] for p in ps))
    end
    res
end

function perturb(p::Particle{T}, scales) where {T}
    n = fieldcount(T)
    res = Array{Any}(undef, 4)
    for i = 1:n
        res[i] = p.x[i] + rand() * scales[i]
    end
    Particle(T(res))
end


# prior is a tuple of N univariate distributions
struct Prior{N} 
    dists::NTuple{N, UnivariateDistribution}
    Prior(args::UnivariateDistribution...) = new{length(args)}(args)
end

Base.length(p::Prior{N}) where {N} = N

function Distributions.pdf(p::Prior{N}, x) where {N}
    res = pdf(p.dists[1], x[1])
    for i = 2:N
        res *= pdf(p.dists[i], x[i])
    end
    res
end

function Distributions.logpdf(p::Prior{N}, x) where {N}
    res = logpdf(p.dists[1], x[1])
    for i = 2:N
        res += logpdf(p.dists[i], x[i])
    end
    res
end

function Distributions.rand(p::Prior{N}) where {N}
    Particle(ntuple(i -> rand(p.dists[i]), N))
end



abstract type ABCAlgorithm end

struct AdaptiveABC <: ABCAlgorithm
    N::Int64       # number of particles to use
    η::Float64     # resample once ESS < N * η
    abs_tol::Float64
    rel_tol::Float64
    α::Float64          # alpha (coverage of ϵ to include in new population)
    M::Int64            # number os simulations to perform per particle

    AdaptiveABC(;N = 100, η = 0.5, abs_tol = 0.0, rel_tol = 0.001, α = 0.95, M = 1) =
        new(N, η, abs_tol, rel_tol, α, M)
end


# Model is a function(params) that simulates data based on params
# and returns distance between simulated and observed data
#
# created by defining make_model function

# effective sample size
function compute_ess(weights)
    res = 0.0
    for i in eachindex(weights)
        res += weights[i]^2
    end
    1/res
end


function compute_weights_epsilon(weights_prev, eps_prev, distances, alpha)

    ess_prev = compute_ess(weights_prev)

    success_prev = sum(distances .<= eps_prev, dims = 2)

    weights = similar(weights_prev)
    
    function f(eps_)
        success = sum(distances .<= eps_, dims = 2)
        weights .= weights_prev .* success ./ success_prev
        weights .= weights ./ sum(weights)
        ess_new = compute_ess(weights)
        alpha * ess_prev - ess_new
    end

    weights, find_zero(f, (minimum(distances) + eps(Float64), eps_prev))

end

function sample(model::Function, prior::Prior{N}, alg::AdaptiveABC) where {N}

    particles = [rand(prior) for i in 1:alg.N]
    distances = Array{Float64}(undef, (alg.N, alg.M))
    #successes = Array{Int64}(undef, alg.N)  # successes[i] = how many sims have model(particle[i]) <= ϵ
    weights = ones(Float64, alg.N) / alg.N
   
    n = 0
    ϵ = Inf

    # sample from prior and compute distances for M simulations per particle 
    for i = 1:alg.N
        #particles[i] = rand(prior)

        for j = 1:alg.M
            distances[i, j] = model(particles[i].x) 
        end
    end

    ess = compute_ess(weights)
    @show n, ϵ, ess
    
    while true
        n = n + 1
        ϵ_prev = ϵ
        weights, ϵ = compute_weights_epsilon(weights, ϵ_prev, distances, alg.α)
        ess = compute_ess(weights)
        @show n, ϵ, ess

        if ess < alg.η * N
            # resample
            idx = wsample(1:alg.N, weights)
            particles .= particles[idx]
            fill!(weights, 1/alg.N)
        end

        scales = 2*std(particles)
        
        for i = 1:alg.N
            particles[i] = perturb(particles[i], scales)
            for j = 1:alg.M
                distances[i, j] = model(particles[i].x) 
            end
        end

        if 2 * abs(ϵ_prev - ϵ) < alg.rel_tol * (abs(ϵ_prev) + abs(ϵ)) ||
            ϵ < alg.abs_tol
            break
        end
        
    end

    (particles = particles,
     weights = weights,
     ϵ = ϵ)
    
end

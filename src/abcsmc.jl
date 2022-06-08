
using Distributions
using Roots
using StatsBase
import Random

export Particle,
    Prior,
    GaussianKernel,
    AdaptiveABC,
    sample


struct Particle{T}
    x::T
    Particle(x::T) where {T} = new{T}(x)
end

Base.length(::Particle{T}) where {T} = fieldcount(T)


# prior is a tuple of N univariate distributions
# for prior we treat all dimensions as independent
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


abstract type Kernel end

mutable struct GaussianKernel <: Kernel
    σ::Vector{Float64}  # let's assume diagonal 
    GaussianKernel(σ = Vector{Float64}()) = new(σ)
end

(gk::GaussianKernel)(p1, p2) = sum(pdf.(Normal.(0.0, gk.σ), p1.x .- p2.x))

function perturb(p::Particle{T}, k::GaussianKernel) where {T} 
    n = fieldcount(T)
    res = Array{Any}(undef, n)
    for i = 1:n
        res[i] = p.x[i] + rand(Normal()) * k.σ[i]
    end
    Particle(T(res))
end

function update!(k::GaussianKernel, ps::Vector{Particle{T}}, ws) where {T}
    # want weighted standard deviation for each dimension
    # provided by StatsBase.std
    n = fieldcount(T)
    res = Array{Float64}(undef, n)
    for i = 1:n
        res[i] = std([p.x[i] for p in ps], pweights(ws))
    end
    k.σ = 2*res
    k
end


abstract type ABCAlgorithm end

struct AdaptiveABC <: ABCAlgorithm
    N::Int64       # number of particles to use
    abs_tol::Float64
    rel_tol::Float64
    α::Float64          # alpha (coverage of ϵ to include in new population)
    M::Int64            # number os simulations to perform per particle
    kernel::Kernel

    AdaptiveABC(;N = 100,
                abs_tol = 0.0, rel_tol = 0.001,
                α = 0.95,
                M = 1,
                kernel = GaussianKernel()) =
        new(N, abs_tol, rel_tol, α, M, kernel)
end


"compute effective sample size ESS"
function compute_ess(weights)
    res = 0.0
    for i in eachindex(weights)
        res += weights[i]^2
    end
    1/res
end


"""
compute ϵ for next iteration using method from Del Moral et al.(2011)

ESS(weights_prev .* successrate(ϵ) ./ succesrate(ϵ_prev)) = α ESS(weights_prev)

"""
function compute_epsilon(weights_prev, eps_prev, distances, alpha)

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

    find_zero(f, (minimum(distances) + eps(Float64), eps_prev))

end


using TimerOutputs
to = TimerOutput()


# Model is a function(params) that simulates data based on params
# and returns distance between simulated and observed data
"""
sample from model using ABC SMC method as described in Toni et al.(2009),
but determine sequence of ϵ adaptively

Calculation of weights is O(N^2)
"""
function sample(model::Function, prior::Prior{N}, alg::AdaptiveABC; verbose = true) where {N}

    particles = [rand(prior) for i in 1:alg.N]
    particles_prev = similar(particles)
    distances = Array{Float64}(undef, (alg.N, alg.M))

    weights = ones(Float64, alg.N) / alg.N
    weights_prev = ones(Float64, alg.N) / alg.N
    
    n = 0
    ϵ = Inf

    # sample from prior and compute distances for M simulations per particle 
    for i = 1:alg.N
        #particles[i] = rand(prior)

        for j = 1:alg.M
            distances[i, j] = model(particles[i].x) 
        end
    end

    ess = alg.N
    
    while true
        
        n = n + 1
        ϵ_prev = ϵ
        particles_prev .= particles
        weights_prev .= weights

        # compute new ϵ
        @timeit to "epsilon" ϵ = compute_epsilon(weights_prev, ϵ_prev, distances, alg.α)
 
        update!(alg.kernel, particles_prev, weights_prev)

        @timeit to "loop" begin
            for i = 1:alg.N
                s = 0  # number of successes
                while s == 0
                    @timeit to "wsample" j = wsample(1:alg.N, weights_prev)
                    @timeit to "perturb" particles[i] = perturb(particles_prev[j], alg.kernel)
                    pdf(prior, particles[i].x) == 0 && continue
                    @timeit to "sim" for j = 1:alg.M
                        distances[i, j] = model(particles[i].x)
                        if distances[i,j] <= ϵ
                            s += 1
                        end
                    end
                end

                @timeit to "weights" begin
                    denominator = 0.0
                    @timeit to "kernel" for j = 1:alg.N
                        denominator += weights_prev[j] * alg.kernel(particles[i], particles_prev[j])
                    end
                    @timeit to "weights" weights[i] = pdf(prior, particles[i].x) * s / denominator
                end
            end
        end
        # normalize weights
        weights .= weights ./ sum(weights)

        ess = compute_ess(weights)

        verbose && println("Iteration $n: ϵ = $ϵ, ess = $ess")
        
        if 2 * abs(ϵ_prev - ϵ) < alg.rel_tol * (abs(ϵ_prev) + abs(ϵ)) ||
            ϵ < alg.abs_tol
            break
        end
        
    end

    (particles = particles,
     weights = weights,
     ϵ = ϵ)
    
end


abstract type AbstractParticle end

mutable struct Particle <: AbstractParticle
    params::Vector{Float64}
    weight::Float64
    distance::Float64
end

abstract type ABCAlgorithm end

struct ABCSMC <: ABCAlgorithm
    n_particles::Int64       # number of particles to use
    eps_start::Float64
    eps_end::Float64
    alpha::Float64           # alpha (coverage of ϵ to include in new population)
    n_sims::Int64            # number os simulations to perform per particle
    max_iters::Int64         # maximum (outer) iterations sampling params 

    ABCSMC(n_particles, eps_start, eps_end, alpha, n_sims, max_iters) =
        new(n_particles, eps_start, eps_end, alpha, n_sims, max_iters)
end


# Model is a function(params) that simulates data based on params
# and returns distance between simulated and observed data
#
# created by defining make_model function




function sample(model::Function, priors, alg::ABCSMC)

    particles = Array{Particle}(undef, alg.n_particles)

    #TODO: for each param, keep minimum distance from inner loop
    #      if we do not get n_particles in max_iters,
    #      then use the particles with lowest distance
    
    # t = 0, sample from prior until n_particles accepted
    ϵ = alg.eps_start
    total_s = 0
    for i in 1:alg.n_particles
        niter = 0
        s = 0
        max_distance = 0.0
        while s == 0
            niter += 1
            params = [rand(d) for d in priors]
            
            for j in 1:alg.n_sims
                distance = model(params)
                if distance <= ϵ
                    s += 1
                    max_distance = (distance > max_distance) ? distance : max_distance 
                end
            end
        end
        particles[i] = Particle(params, s, max_distance)
        total_s += s
    end

    # normalize weights
    for i in 1:alg.n_particles
        particles[i].weight /= total_s
    end
    
        
    
end

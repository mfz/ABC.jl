# Toy example from section 4.1 in Del Moral et al.(2011)

using ABC
using Distributions
using Plots
using TimerOutputs

"function to simulate data set given parameters"
function toymodel(params)
    Θ = params[1]
    q = rand()
    x = q < 0.5 ? rand(Normal(Θ, 1)) : rand(Normal(Θ, 0.1))
    x
end


"""
create function(params) that computes distance
between observed and simulated data sets
"""
function distance(observed)
    function(params)
        sim = toymodel(params)
        abs(sim - observed)
    end
end

prior = Prior(Uniform(-10,10))

reset_timer!(ABC.to)

res = ABC.sample(distance(0), prior,
             AdaptiveABC(α = 0.9, N = 1000, M = 5,
                         abs_tol = 0.025))

show(ABC.to)

histogram([p.x for p in res.particles], weights = res.weights, bins = 100)


using Profile
@profile res = ABC.sample(distance(0), prior,
             AdaptiveABC(α = 0.9, N = 100, M = 5,
                         abs_tol = 0.025))

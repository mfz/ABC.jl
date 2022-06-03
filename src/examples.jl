# Toy example from section 4.1

using ABC

function toymodel(params)
    Θ = params[1]
    x = 0.5 * rand(Normal(Θ, 1)) + 0.5 * rand(Normal(Θ, 0.01))
    abs(x)
end


alg = AdaptiveABC()

model = toymodel
prior = Prior(Uniform(-10, 10))
i = 1

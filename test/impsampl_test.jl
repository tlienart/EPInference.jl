using Base.Test, EPInference, ExpFamily

# Proposing samples from a wide spherical gaussian.

srand(123)

C  = [1.0 0.5; 0.5 1.0]
P  = 2/3 * [2 -1; -1 2]
m  = [1.0;-0.5]

gNP     = GaussianNatParam(mean=m, cov=C)
gNPprop = ones(GaussianNatParam, 2)/20 # division INFLATES the variance

(samples,w) = impsampling(x->loglik(gNP,x), gNPprop, 10000)

@test isapprox(sum(samples[:,i]*w[i] for i in 1:length(w)), m, rtol=0.05)

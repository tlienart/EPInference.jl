export
    ParamsEP,
    epNP,
    epdNP,
    epMP

immutable ParamsEP
    prior::EFamily
    logfactors::Vector{Function} # NOTE these functions need to be able to handle matrices of size pxN
    nEP::Int
    nIS::Int    # NOTE would have to generalise this --> could be SAMPLER obj
    damp::Float # NOTE would have to generalise this (stepping)
    minvar::Float
    # alpha::Float
    # ------------
    dim::Int
    nfactors::Int
    function ParamsEP(prior,factors,nEP,nIS,damp=0.1,minvar=1e-5)
        @assert nIS>1 "Need at least two samples per estimator"
        #@assert 0.0<damp<=1.0 "Damping parameter should be in (0,1]"
        new(prior, factors, nEP, nIS, damp, minvar,
            length(mean(prior)), length(factors))
    end
end

"""
    epNP

Serial EP updates with damping in the Natural Parameter space.
"""
function epNP(p::ParamsEP)
    globapprox_mem = []

    locapprox  = [ones(GaussianNatParam, p.dim)/100 for i in 1:p.nfactors]
    globapprox = p.prior + sum(locapprox)

    push!(globapprox_mem, globapprox)

    for iter in 1:p.nEP
        for i in randperm(p.nfactors)
            cavity      = globapprox - locapprox[i]
            lltilted(x) = p.logfactors[i](x) + uloglik(cavity, x)
            (iss,w)     = impsampling(lltilted, globapprox, p.nIS)
            muhat       = sum( w[j]*suffstats(GaussianNatParam, iss[:,j])
                                for j in 1:p.nIS )
            # Cast it as mean parameter
            muhat = GaussianMeanParam(muhat[1], muhat[2])
            # Transform to natural parameter, project in case out of NP space
            np_muhat     = natparam(project(muhat,minvar=p.minvar))
            globapprox   = (1.0-p.damp)*globapprox + p.damp*np_muhat
            locapprox[i] = globapprox-cavity
        end
        push!(globapprox_mem, globapprox)
    end
    (globapprox, globapprox_mem)
end

"""
    epNP

Serial EP updates with damping in the Natural Parameter space.
"""
function epdNP(p::ParamsEP)
    locapprox  = [ones(DiagGaussianNatParam, p.dim)/100 for i in 1:p.nfactors]
    globapprox = p.prior + sum(locapprox)
    updatecounter = 0
    for iter in 1:p.nEP
        for i in randperm(p.nfactors)
            cavity      = globapprox - locapprox[i]
            lltilted(x) = p.logfactors[i](x) + uloglik(cavity, x)
            (iss,w)     = impsampling(lltilted, globapprox, p.nIS)
            muhat       = sum( w[j]*suffstats(DiagGaussianNatParam, iss[:,j])
                                for j in 1:p.nIS )
            # Cast it as mean parameter
            muhat = DiagGaussianMeanParam(muhat[1], muhat[2])
            # Transform to natural parameter, project in case out of NP space
            np_muhat = natparam(muhat)
            # Perform the tentative damping step
            t_globapprox = (1.0-p.damp)*globapprox+p.damp*np_muhat
            if maximum(-t_globapprox.theta2) < 1.0/p.minvar
                globapprox   = t_globapprox
                locapprox[i] = globapprox-cavity
                updatecounter += 1
            end
        end
    end
    println("Ratio of accepted updates: $updatecounter/$(p.nEP*p.nfactors)")
    globapprox
end

"""
    epMP

Serial EP updates with damping in the Mean Parameter space.
"""
function epMP(p::ParamsEP)
    globapprox_mem = []

    locapprox  = [ones(GaussianNatParam, p.dim)/100 for i in 1:p.nfactors]
    globapprox = p.prior + sum(locapprox)

    push!(globapprox_mem, globapprox)

    for iter in 1:p.nEP
        for i in randperm(p.nfactors)
            cavity      = globapprox - locapprox[i]
            lltilted(x) = p.logfactors[i](x) + uloglik(cavity, x)
            (iss,w)     = impsampling(lltilted, globapprox, p.nIS)
            muhat       = sum( w[j]*suffstats(GaussianNatParam, iss[:,j])
                                for j in 1: p.nIS )
            globapprox = natparam( (1.0-p.damp)*meanparam(globapprox)+
                                    p.damp*GaussianMeanParam(muhat[1], muhat[2])
                                  )
            locapprox[i] = globapprox-cavity
        end
        push!(globapprox_mem, globapprox)
    end
    (globapprox, globapprox_mem)
end

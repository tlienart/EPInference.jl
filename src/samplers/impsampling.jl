function impsampling(lltarget::Function, proposal::EFamily, nsamples::Int;
                     nattempts=5)::Tuple{Matrix{Float},Vector{Float}}
    attempt   = 1
    notvalid  = true
    samples,w = [],[]
    while notvalid && attempt <= nattempts
        samples       = rand( proposal, nsamples )
        logliksamples = loglik( proposal, samples)

        # IS weights
        lW  = lltarget( samples ) - logliksamples
        lW -= maximum(lW) # avoid underflow
        w   = exp(lW)
        w  /= sum(w)

        notvalid = any(isnan.(w))
        attempt += 1
    end
    if attempt == nattempts+1
        throw("Could not generate a sample in $nattempts attempts")
    end
    (samples, w)
end

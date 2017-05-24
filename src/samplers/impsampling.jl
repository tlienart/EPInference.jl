function impsampling(lltarget::Function, proposal::EFamily, nsamples::Int;
                     nattempts=5)::Tuple{Matrix{Float},Vector{Float}}
    attempt = 1
    while attempt <= nattempts
        samples       = rand(proposal, nsamples)
#        logliksamples = loglikelihood(proposal, samples)

        attempt += 1
    end
    if attempt == nattempts+1
        throw("Could not generate a sample in $nattempts attempts")
    end
    (samples,weights)
end

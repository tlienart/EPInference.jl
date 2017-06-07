module EPInference

using Compat
using ExpFamily

const Float = Float64

export
    impsampling

include("samplers.jl")
include("epalgorithms.jl")

end # module

using CSV
using DataFrames
using LinearAlgebra
using Plots
using Ripserer

function readSim(fileName::String, d, k)
    df = DataFrame(CSV.File(fileName))
    df2 = Tuple.(eachrow(df))

    #print(cocycle)
    diagram_cycles = ripserer(df2; alg=:involuted, dim_max = 2)
    
    n = size(diagram_cycles[d])[1]
    if k < n
        n = k
    end
    
    ans = []
    
    for i in n:-1:1
        most_persistent_ho = diagram_cycles[d][end + 1 - i]
        cycle = representative(most_persistent_ho)
        l = []
        for i in cycle
            push!(l, vertices(i)[1])
            push!(l, vertices(i)[2])
            push!(l, vertices(i)[3])
        end
        push!(ans,l)
    end
    return ans
end


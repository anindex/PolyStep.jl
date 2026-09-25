include("bbob_mixint.jl")

const GRID = vec([(; epsilon, rfrac, decay)
                  for epsilon in (0.03, 0.1, 0.3), rfrac in (0.5, 1.0), decay in (0.95, 0.99)])

function tune(; options = "dimensions: 20 instance_indices: 1-5", mult = 1000)
    cd(@__DIR__)
    scores = map(enumerate(GRID)) do (i, c)
        dir, _ = experiment("tune_$i", options, mult; c...)
        s = auc(runtimes(dir), mult)
        println(c, " auc = ", round(s; digits = 4))
        s
    end
    println("best: ", GRID[argmax(scores)])
    return GRID, scores
end

abspath(PROGRAM_FILE) == @__FILE__() && tune()

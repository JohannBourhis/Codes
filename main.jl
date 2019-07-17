60.0# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, SolverTools

#Benchmark
using SolverTools, SolverBenchmark

# Unconstrained solvers
include("C:/Users/Johann/Documents/BFGS/Test_Problems/trunk.jl")
include("C:/Users/Johann/Documents/BFGS/Test_Problems/test_problems.jl")

scale = true
nlp, M = fletchcr(1000), [1, 3, 5, 10, 15, 20, 100, 1000]
#nlp, M = nondquar(1000), [1, 3, 5, 10, 15, 20, 1000]
solver_basic_1(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1, bk_max =  0,
    memory_bound = -1.0, scale = false)
solver_scaling_1(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1, bk_max =  0,
    memory_bound = -1.0, scale = true)
solver_bound_1(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1, bk_max =  0,
    memory_bound = 1.0, scale = false)
solver_basic_3(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 3, bk_max =  0,
        memory_bound = -1.0, scale = false)
solver_scaling_3(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 3, bk_max =  0,
        memory_bound = -1.0, scale = true)
solver_bound_3(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 3, bk_max =  0,
        memory_bound = 5.0e-02, scale = false)
solver_basic_5(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 5, bk_max =  0,
            memory_bound = -1.0, scale = false)
solver_scaling_5(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 5, bk_max =  0,
            memory_bound = -1.0, scale = true)
solver_bound_5(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 5, bk_max =  0,
            memory_bound = 1.0, scale = false)
solver_basic_10(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 10, bk_max =  0,
                memory_bound = -1.0, scale = false)
solver_scaling_10(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 10, bk_max =  0,
                memory_bound = -1.0, scale = true)
solver_bound_10(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 10, bk_max =  0,
                memory_bound = 5.0e-02, scale = false)
solver_basic_15(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 15, bk_max =  0,
                                memory_bound = -1.0, scale = false)
solver_scaling_15(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 15, bk_max =  0,
                                memory_bound = -1.0, scale = true)
solver_bound_15(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 15, bk_max =  0,
                                memory_bound = 5.0e-01, scale = false)
solver_basic_20(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 20, bk_max =  0,
                                memory_bound = -1.0, scale = false)
solver_scaling_20(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 20, bk_max =  0,
                                    memory_bound = -1.0, scale = true)
solver_bound_20(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 20, bk_max =  0,
                                    memory_bound = 5.0e-02, scale = false)
solver_basic_100(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 100, bk_max =  0,
    memory_bound = -1.0, scale = false)
solver_scaling_100(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 100, bk_max =  0,
    memory_bound = -1.0, scale = true)
solver_bound_100(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 100, bk_max =  0,
    memory_bound = 5.0e-01, scale = false)
solver_basic_1000(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1000, bk_max =  0,
    memory_bound = -1.0, scale = false)
solver_scaling_1000(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1000, bk_max =  0,
    memory_bound = -1.0, scale = true)
solver_bound_1000(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1000, bk_max =  0,
    memory_bound = 5.0e-01, scale = false)
solver_CG(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1000, bk_max =  0,
    memory_bound = -1.0, CG = true)

"""
solver_list = Dict(:Base1 => solver_basic_1, #:Bound1 => solver_bound_1,
                :Base3 => solver_basic_3, #:Bound3 => solver_bound_3,
                :Base5 => solver_basic_5, #:Bound5 => solver_bound_5,
                :Base10 => solver_basic_10, #:Bound10 => solver_bound_10,
                :Base15 => solver_basic_15, #:Bound15 => solver_bound_15,
                :Base20 => solver_basic_20, #:Bound20 => solver_bound_20,
                :Base100 => solver_basic_100, #:Bound100 => solver_bound_100,
                :Base1000 => solver_basic_1000, #:Bound1000 => solver_bound_1000,
                :CG => solver_CG)
"""

solver_list = Dict(:Bound5 => solver_bound_5,
                    :Bound1 => solver_bound_1)
                    #:Base100 => solver_bound_1000)
                    #:CG => solver_CG)

prob_1 = fletchcr(1000)
prob_2 = nondquar(1000)
prob_3 = woods(1000)
#prob_4 = noncvxun(1000)
prob_5 = sparsine(1000)
problem_list = [prob_1, prob_2, prob_3, prob_5]

stats = bmark_solvers(solver_list, problem_list)

"""
tables = ["Bound1.tex", "Base1.tex", "Bound3.tex", "Base3.tex", "Bound5.tex", "Base5.tex",
            "Bound10.tex", "Base10.tex", "Bound15.tex", "Base15.tex",
            "Bound20.tex", "Base20.tex", "Bound100.tex", "Base100.tex",
            "Bound1000.tex", "Base1000.tex", "CG.tex"]
key = [:Bound1, :Base1, :Bound3, :Base3, :Bound5, :Base5,
            :Bound10, :Base10, :Bound15, :Base15,
            :Bound20, :Base20, :Bound100, :Base100,
            :Bound1000, :Base1000, :CG]
"""
tables = ["Bound5.tex", "Bound1.tex"]

key = [:Bound5, :Bound1]

for i in 1:length(key)
    df = get(stats, key[i], false)
    df1 = df[:, [2, 6, 7, 8, 9, 12, 13, 21]]
    open(tables[i], "w") do io
        latex_table(io, df1)
    end
end

"""
for m in M
    println("____________")
    println("memory = ", m)
    stats = trunk(nlp, max_time = 120.0, nm_itmax = 400, memory = m, bk_max =  0,
        memory_bound = -1.0, scale = scale);
    println("neval_obj = ", nlp.counters.neval_obj)
    println("neval_grad = ", nlp.counters.neval_grad)
    println("neval_hprod = ", nlp.counters.neval_hprod)
    print(stats)
    reset!(nlp)
end
"""

"""
for m in M
    println("____________")
    println("memory = ", m, "+")
    stats = trunk(nlp, max_time = 60.0, nm_itmax = 400, memory = m, bk_max =  0, memory_bound = 5.0e-01)
    println("neval_obj = ", nlp.counters.neval_obj)
    println("neval_grad = ", nlp.counters.neval_grad)
    println("neval_hprod = ", nlp.counters.neval_hprod)
    print(stats)
    reset!(nlp)
end
"""

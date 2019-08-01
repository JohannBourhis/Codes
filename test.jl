# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, SolverTools

#Benchmark
using SolverTools, SolverBenchmark

# Unconstrained solvers
include("trunk.jl")
include("test_problems.jl")

scale = true
prob = NZF1(1000)
#nlp, M = nondquar(1000), [1, 3, 5, 10, 15, 20, 1000]
"""
trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1, bk_max =  0,
    memory_bound = -1.0, scale = false)
trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 5, bk_max =  0,
    memory_bound = -1.0, scale = false)
"""
stats = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1, bk_max =  0,
    memory_bound = -1.0, scale = false)

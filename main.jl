60.0# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, SolverTools

#Benchmark
using SolverTools, SolverBenchmark

# Unconstrained solvers
include("C:/Users/Johann/Documents/BFGS/Test_Problems/trunk.jl")
include("C:/Users/Johann/Documents/BFGS/Test_Problems/test_problems.jl")

solver_1(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 1, bk_max =  0,
    memory_bound = -1.0, scale = false)
solver_3(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 3, bk_max =  0,
    memory_bound = -1.0, scale = false)
solver_5(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 5, bk_max =  0,
    memory_bound = -1.0, scale = false)
solver_10(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 10, bk_max =  0,
    memory_bound = -1.0, scale = false)
solver_15(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 15, bk_max =  0,
    memory_bound = -1.0, scale = false)
solver_20(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 20, bk_max =  0,
    memory_bound = -1.0, scale = false)
solver_CG(prob) = trunk(prob, max_time = 60.0, nm_itmax = 400, memory = 20, bk_max =  0,
    memory_bound = -1.0, scale = false, CG = true)


solver_list = Dict(:S1 => solver_1,
                    :S3 => solver_3,
                    :S5 => solver_5,
                    :S10 => solver_10,
                    :S15 => solver_15,
                    :S20 => solver_20,
                    :CG => solver_CG)

prob_1 = arglina()
prob_2 = arglinb()
prob_3 = arglinc()
prob_4 = arwhead()
prob_5 = bdqrtic()
prob_6 = beale()
prob_7 = broydn7d()
prob_8 = brybnd()
prob_9 = chainwoo()
prob_10 = chnrosnb_mod()
prob_11 = cosine()
prob_12 = cragglvy()
prob_13 = dixmaane()
prob_14 = dixmaani()
prob_15 = dixmaanm()
prob_16 = dixon3dq()
prob_17 = dqdrtic()
prob_18 = dqrtic()
prob_19 = edensch()
prob_20 = eg2()
prob_21 = engval1()
prob_22 = errinros_mod()
prob_23 = extrosnb()
prob_24 = fletcbv2()
prob_25 = fletcbv3_mod()
prob_26 = fletchcr()
prob_27 = freuroth()
prob_28 = genhumps()
prob_29 = genrose()
prob_30 = genrose_nash()
prob_31 = indef_mod()
prob_32 = liarwhd()
prob_33 = morebv()
prob_34 = ncb20()
prob_35 = ncb20b()
prob_36 = noncvxu2()
prob_37 = noncvxun()
prob_38 = nondia()
prob_39 = nondquar()
prob_40 = NZF1()
prob_41 = penalty2()
prob_42 = penalty3()
prob_43 = powellsg()
prob_44 = power()
prob_45 = quartc()
prob_46 = sbrybnd()
prob_47 = schmvett()
prob_48 = scosine()
prob_49 = sparsine()
prob_50 = sparsqur()
prob_51 = srosenbr()
prob_52 = sinquad()
prob_53 = tointgss()
prob_54 = tquartic()
prob_55 = tridia()
prob_56 = vardim()
prob_57 = woods()

problem_list = [prob_1, prob_2,prob_3,prob_4,prob_5,prob_6,prob_7,prob_8,prob_9,
                prob_10,prob_11,prob_12,prob_13,prob_14,prob_15,prob_16,prob_17,
                prob_18,prob_19,prob_20,prob_21,prob_22,prob_23,prob_24,prob_25,
                prob_26,prob_27,prob_28,prob_29,prob_30,prob_31,prob_32,prob_33,
                prob_34,prob_35,prob_36,prob_37,prob_38,prob_39,prob_40,prob_41,
                prob_42,prob_43,prob_44,prob_45,prob_46,prob_47,prob_48,prob_49,
                prob_50,prob_51,prob_52,prob_53,prob_54,prob_55,prob_56,prob_57]

stats = bmark_solvers(solver_list, problem_list)

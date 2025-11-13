using NonlinearSolve
using BenchmarkTools
using StaticArrays

f_SA(u,p)=u.*u.-p
u0=SA[1.0,1.0]
p=2.0
prob=NonlinearProblem(f_SA,u0,p)
@benchmark solve(prob,NewtonRaphson())

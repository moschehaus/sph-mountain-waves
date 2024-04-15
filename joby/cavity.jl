import Pkg
Pkg.add("Printf")
Pkg.add("SmoothedParticles")
Pkg.add("Plots")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("LaTeXStrings")
Pkg.add("Parameters")

include("cavity.flow")
cavity_flow.main()

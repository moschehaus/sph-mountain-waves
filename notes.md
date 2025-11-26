## Adaptive smoothing length
- sources:
    - high density ratios (for the first approach)
    - magnetohydrodynamics (for the second approach)
- two approaches:
    - when the density is solved for by the continuity equation, you can also solve for the smoothing length
    - in SmoothedParticles.jl you however need to pay attention how you use the `apply!` operator: you must call it in its unary version, i.e. *do not call it inside* `balance_of_mass!(p,q,r)`
    - when the density is calculated after the computations via the kernel interpolation, you must solve a *nonlinear problem*, as when you are calculating the density, you are using the kernel, which uses the smoothing length which is however a function of the density itself

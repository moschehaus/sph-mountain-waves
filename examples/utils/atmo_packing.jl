#Implementation of the packing algorithm for the witch. jl simulationfrom the aritcle: "Particle packing algorithm for SPH schemes, doi: 10.1016/j.cpc.2012.02.032"#
# 
module atmo_packing

using SmoothedParticles
using Statistics
using LinearAlgebra


#geometry parameters
const dom_height = 26e3   #height of the domain 
const dr = dom_height/150  #average particle distance (decrease to make finer simulation)
const h = 1.8*dr              

#physical parameters
const rho0 =1.393		 #referential fluid density

#meteorological parameters
const g=9.81
const R_mass=287.05
const T=250
const c = sqrt(65e3*(7/5)/rho0)	 #numerical speed of sound

#temporal parameters
const dt = 0.01*h/c                #time step

#particle types
const FLUID = 0.0
const INFLOW = 1.0
const WALL = 2.0
const MOUNTAIN = 3.0

"""
    packing(system::ParticleSystem,abs_tol::Float64,rel_tol::Float64,maxSteps::Int64)
For the given system, abs_tol, rel_tol, maxSteps finds a stable initial condition such that the velocities and pressure gradients
are lesser then a stopping criterion based on abs_tol, rel_tol. If the criterion is not met, ends after maxSteps iterations.

"""
function packing(system::ParticleSystem,abs_tol::Float64,rel_tol::Float64,maxSteps::Int64)
    particles=system.particles
    β,ζ,V0=get_packing_pars(system)
    apply!(system, (p, q,r) -> find_gGamma!(p, q, r, V0))
    
    init_vel=[]
    init_gGamma=[]
    for par in particles
        u=par.u
        ∇Γ=par.gGamma
        push!(init_vel,u)
        push!(init_gGamma,∇Γ)
    end
    res_vel=LinearAlgebra.norm(init_vel)
    res_gGamma=LinearAlgebra.norm(init_gGamma)
    println("-----------------PACKING ALGORITHM INITIALIZED---------------")
    println("Initial norms of velocity and ∇Γ are $res_vel and $res_gGamma")
    numSteps=0
    while (((res_vel+res_gGamma)>=stopping_criterion(abs_tol,rel_tol,init_vel,init_gGamma)) && numSteps<maxSteps)
        apply!(system, packing_accelerate!)
        apply!(system, packing_move!)
        create_cell_list!(system)
        apply!(system, (p, q,r) -> find_gGamma!(p, q, r, V0))
        apply!(system, p->stabilization_force!(p,β,ζ))
        apply!(system, packing_accelerate!)

        velocities=[]
        gGamma=[]
        for par in particles
            u=par.u
            ∇Γ=par.gGamma
            push!(velocities,u)
            push!(gGamma,∇Γ)
        end
        res_vel=LinearAlgebra.norm(velocities)
        res_gGamma=LinearAlgebra.norm(gGamma)
        numSteps+=1
    end

    if numSteps < maxSteps
        println("Packing successful after $numSteps iterations with velocities norm $res_vel and ∇Γ norm $res_gGamma")
    else
        println("Packing unsuccessful, maximum number of iterations reached, velocities norm $res_vel and ∇Γ norm $res_gGamma")
    end
    println("-----------------PACKING ALGORITHM FINISHED---------------")
    apply!(system,reset!)
    return system
end


@inbounds function stabilization_force!(p::AbstractParticle,β::Float64,ζ::Float64)
    p.Du=-β * p.gGamma- ζ * p.u
    p.gGamma=VEC0
end

@inbounds function find_gGamma!(p::AbstractParticle, q::AbstractParticle,r::Float64,V0::Float64)
    x_pq=p.x-q.x
    p.gGamma += V0*rDwendland2(h,r)*x_pq    
end

function get_packing_pars(system::ParticleSystem)
    
    particles=system.particles
    
    K = g / (T * R_mass)
    ymin=0.0
    ymax=dom_height
    p0=(rho0^2 * T^2 * R_mass^2 / g) * (exp(-K * ymin) - exp(-K * ymax)) #average pressure
    ρ0=(rho0 * T * R_mass / g) * (exp(-K * ymin) - exp(-K * ymax)) #average density
    
    volume=[]
    for par in particles
        vol=par.m/par.rho
        push!(volume,vol)
    end
    V0=Statistics.mean(volume) #average volume
    
    α=5e-3 #viscosity parameter
    β=2*p0/ρ0
    ζ=α*sqrt(β/V0)

    return β, ζ, V0
end

function packing_accelerate!(p::AbstractParticle)
    if p.type==FLUID
	    p.u += 0.5*dt*p.Du
    else
        p.u=VEC0
    end
end

function packing_move!(p::AbstractParticle)
	p.Du = VEC0
	if p.type == FLUID
		p.x += dt*p.u
	end

end

function reset!(p::AbstractParticle)
    p.Du=VEC0
    p.u=VEC0
end

function stopping_criterion(abs_tol::Float64,rel_tol::Float64,init_vel::Vector{Any},init_gGamma::Vector{Any})
    return 2*abs_tol+rel_tol*(LinearAlgebra.norm(init_gGamma)+LinearAlgebra.norm(init_vel))
end

end
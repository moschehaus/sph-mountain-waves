#Implementation of the packing algorithm for the witch simulations from the aritcle: "Particle packing algorithm for SPH schemes, doi: 10.1016/j.cpc.2012.02.032"#
# 
using Statistics
using LinearAlgebra


"""
    packing(system::ParticleSystem,abs_tol::Float64,rel_tol::Float64,maxSteps::Int64)
For the given system, abs_tol, rel_tol, maxSteps finds a stable initial condition such that the velocities and pressure gradients
are lesser then a stopping criterion based on abs_tol, rel_tol. If the criterion is not met, ends after maxSteps iterations.

"""
function packing(system::ParticleSystem, abs_tol::Float64, rel_tol::Float64, maxSteps::Int64)
        apply!(system, reset!) #hard reset of all velocities and accelerations
        particles = system.particles
        β, ζ, V0 = get_packing_pars(system)
        apply!(system, (p, q, r) -> find_gGamma!(p, q, r, V0), self=true)
        apply!(system, p -> stabilization_force!(p, β, ζ))

        init_gGamma = []
        for par in particles
                ∇Γ = par.gGamma
                push!(init_gGamma, ∇Γ)
        end
        res_vel = 0.0 #velocity residuum can be initialized as zero as it has been reset
        res_gGamma = LinearAlgebra.norm(init_gGamma)
        println("-----------------PACKING ALGORITHM INITIALIZED---------------")
        println("Initial norm of ∇Γ is $res_gGamma")
        numSteps = 0
        while (((res_vel + res_gGamma) >= stopping_criterion(abs_tol, rel_tol, init_gGamma)) && numSteps < maxSteps)
                apply!(system, packing_accelerate!)
                apply!(system, packing_move!)
                create_cell_list!(system)
                apply!(system, (p, q, r) -> find_gGamma!(p, q, r, V0), self=true)
                apply!(system, p -> stabilization_force!(p, β, ζ))
                apply!(system, packing_accelerate!)

                velocities = []
                gGamma = []
                for par in particles
                        v = par.v
                        ∇Γ = par.gGamma
                        push!(velocities, v)
                        push!(gGamma, ∇Γ)
                end
                res_vel = LinearAlgebra.norm(velocities)
                res_gGamma = LinearAlgebra.norm(gGamma)
                numSteps += 1
        end

        if numSteps < maxSteps
                println("Packing successful after $numSteps iterations with velocities norm $res_vel and ∇Γ norm $res_gGamma. Velocities will be set to zero now.")
        else
                println("Packing unsuccessful, maximum number of iterations reached, velocities norm $res_vel and ∇Γ norm $res_gGamma. Velocities will be set to zero now")
        end
        println("-----------------PACKING ALGORITHM FINISHED---------------")
        apply!(system, reset!) #hard reset of all velocities and accelerations
        return system
end

#=
Calculate the stabilization force. It consists of a viscous term and a gradient of "uneveness" of distr. of particles
=#

@inbounds function stabilization_force!(p::AbstractParticle, β::Float64, ζ::Float64)
        p.Dv = -β * p.gGamma - ζ * p.v
end

#=
Calculate the graident of gamma, a measure of "uneveness" od particle distribution
=#

@inbounds function find_gGamma!(p::AbstractParticle, q::AbstractParticle, r::Float64, V0::Float64)
        x_pq = p.x - q.x
	p.gGamma += V0 * rDwendland2(p.h, r) * x_pq
end

#=
Get the parameters for the stabilization force: average density, volume and pressure
=#

function get_packing_pars(system::ParticleSystem)

        particles = system.particles

        K = g / (T0 * R_mass)
        ymin = 0.0
        ymax = dom_height
        p0 = (rho0^2 * T0^2 * R_mass^2 / g) * (exp(-K * ymin) - exp(-K * ymax)) #average pressure
        ρ0 = (rho0 * T0 * R_mass / g) * (exp(-K * ymin) - exp(-K * ymax)) #average density

        volume = []
        for par in particles
                vol = par.m / par.rho
                push!(volume, vol)
        end
        V0 = Statistics.mean(volume) #average volume

        #α = 5e-3 #viscosity parameter
        #β = 2 * p0 / ρ0
        ζ = α * sqrt(β / V0)

        return β, ζ, V0
end

#=
Move and accelerate
=#

function packing_accelerate!(p::AbstractParticle)
        if p.type == FLUID
                p.v += 0.5 * dt * p.Dv
        end
end

function packing_move!(p::AbstractParticle)
        p.Dv = VEC0
        p.gGamma = VEC0
        if p.type == FLUID
                p.x += dt * p.v
        end

end
#=
Hard reset of velocities and accelerations, makes sure the initial state is properly set.
=#

function reset!(p::AbstractParticle)
        p.Dv = VEC0
        p.v = VEC0
end

#= 
Simple stopping criterion, combining absolute and relative tolerance
=#

function stopping_criterion(abs_tol::Float64, rel_tol::Float64, init_gGamma::Vector{Any})
        return 2 * abs_tol + rel_tol * (LinearAlgebra.norm(init_gGamma))
end


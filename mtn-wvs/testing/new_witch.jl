"""
 Static atmosphere above a mountain with the Witch of Agnesi profile:

 h(x) = (hₘ a²) / (x² + a²),

 all thermodynamic processes are adiabatic
"""

module AdiabaticStaticWitch
export main

using Printf
using SmoothedParticles
using DataFrames
using Plots
include("../../examples/utils/atmo_packing.jl")
using .atmo_packing

const folder_name = "../results/adiabatic_static_witch"
const export_vars = (:v, :rho, :P, :θ, :T, :type)

# ==============
# Constants
# ==============

# geometry parameters
const dom_height = 26e3        # height of the domain
const dom_length = 400e3       # length of the domain
const dr = dom_height / 50     # average particle distance
const h0 = 1.8 * dr            # smoothing length
const bc_width = 6 * dr        # boundary width
const hₘ = 0.0                 # Witch of Agnesi: mountain height
const a = 0.0                  # Witch of Agnesi: mountain width

# physical parameters
const rho0 = 1.393             # reference density
const mu = 1.0                 # dynamic viscosity
const c = sqrt(65e3 * (7 / 5) / rho0) # speed of sound
const nu = 0.1 * h0 * c

# meteorological parameters
const N = sqrt(0.0196)         # Brunt–Väisälä frequency
const g = 9.81                 # gravity
const R_mass = 287.05          # specific gas constant
const γᵣ = 10 * N              # damping coefficient
const zᵦ = 12e3                # bottom of damping layer
const zₜ = dom_height          # top of damping layer

# thermodynamical parameters
const R_gas = 8.314            # universal gas constant
const cp = 7 * R_mass / 2
const cv = cp - R_mass
const γ = cp / cv              # Poisson constant
const T0 = 250.0               # initial temperature

# temporal parameters
const dt = 0.01 * h0 / c       # time step
const t_end = 5.0              # end of simulation
const dt_frame = t_end / 100   # frame interval

# particle types
const FLUID = 0.0
const WALL = 1.0
const MOUNTAIN = 2.0

# numerical safety floors
const rho_floor = 1e-6
const P_floor = 1e-10

# ==============
# Particle type
# ==============

mutable struct Particle <: AbstractParticle
        h::Float64        # smoothing length
        Dh::Float64       # rate of smoothing length
        x::RealVector     # position
        m::Float64        # mass
        v::RealVector     # velocity
        Dv::RealVector    # acceleration
        rho::Float64      # density
        Drho::Float64     # rate of density
        P::Float64        # kernel-averaged pressure (for output)
        θ::Float64        # potential temperature
        T::Float64        # temperature
        gGamma::RealVector
        type::Float64     # particle type
        A::Float64        # Hopkins entropy variable

        function Particle(x::RealVector, v::RealVector, type::Float64)
                obj = new(
                        h0, 0.0,           # h, Dh
                        x, 0.0,            # x, m
                        v, VEC0,           # u, Dv
                        0.0, 0.0,          # rho, Drho
                        0.0, 0.0,          # P, θ, 
                        0.0,               # T
                        VEC0,              # gGamma
                        type,              # type
                        0.0,               # A, P
                )

                # initial hydrostatic isothermal state 
                obj.T = T0
                obj.rho = rho0 * exp(-obj.x[2] * g / (R_mass * obj.T))
                obj.m = obj.rho * dr^2
                obj.P = R_mass * obj.T * obj.rho
                obj.θ = obj.T * (((T0 * R_gas * rho0) / obj.P)^(2))^(1 / 7)
                obj.A = obj.P / obj.rho^γ
                return obj
        end
end

# ==============
# Geometry & system construction
# ==============

function make_system()
        grid = Grid(dr, :hexagonal)
        domain = Rectangle(-dom_length / 2.0, 0.0, dom_length / 2.0, dom_height)
        fence = BoundaryLayer(domain, grid, bc_width)

        witch_profile(x) = (hₘ * a^2) / (x^2 + a^2)
        mountain = Specification(domain, x -> (x[2] <= witch_profile(x[1])))

        sys = ParticleSystem(Particle, domain + fence, h0)
        generate_particles!(sys, grid, domain - mountain, x -> Particle(x, VEC0, FLUID))
        generate_particles!(sys, grid, fence, x -> Particle(x, VEC0, WALL))
        generate_particles!(sys, grid, mountain, x -> Particle(x, VEC0, FLUID))

        create_cell_list!(sys)
        improved_sys = sys#atmo_packing.packing(sys, 1e-10, 1e-10, 100)
        create_cell_list!(improved_sys)
        return improved_sys
end

# ==============
# Thermodynamics helper: A and P
# ==============

@inbounds function compute_pressure_bar!(p::Particle, q::Particle, r::Float64)
        if p.type == FLUID
                p.P = 0.0
                ker = wendland2(0.5 * (p.h + q.h), r)
                p.P += q.m * q.A^(1 / γ) * ker
        end
end

@inbounds function finalize_pressure_bar!(p::Particle)
        if p.type == FLUID
                p.P = p.P^γ
        end
end

# ==============
# Thermodynamics (simplified for P–A)
# ==============

@inbounds function find_temperature!(p::Particle)
        if p.type == FLUID
                p.T = p.P / (R_mass * p.rho)
        end
end

@inbounds function find_pot_temp!(p::Particle)
        if p.type == FLUID
                p.θ = p.T * (((T0 * R_gas * rho0) / p.P)^(2))^(1 / 7)
        end
end

# ==============
# Smoothing-length & density evolution
# ==============

@inbounds function compute_density!(p::Particle, q::Particle, r::Float64)
        if p.type == FLUID 
                p.rho = 0.0
                p.rho += q.m * wendland(p.h, r)
        end
end

@inbounds function update_smoothing!(p::Particle)
        if p.type == FLUID
                p.h += dt * p.Dh
        end
        p.Dh = 0.0
end


# ==============
# Rayleigh damping
# ==============

function damping_structure(z, zₜ, zᵦ, γᵣ)
        # currently disabled
        return 0.0
end

# ==============
# P–A momentum equation 
# ==============

@inbounds function balance_of_momentum!(p::Particle, q::Particle, r::Float64)
        x_pq = p.x - q.x

        if p.type == FLUID
                # Hopkins pressure–entropy prefactor (Aᵢ Aⱼ)^{1/γ}
                prefac = q.m * (p.A * q.A)^(1 / γ)

                # P̄ exponents P̄^{1 - 2/γ}
                expfac = 1.0 - 2.0 / γ

                # use reduced derivative kernels with each particle's h
                ker_i = rDwendland2(p.h, r)
                ker_j = rDwendland2(q.h, r)

                # pairwise conservative force
                p.Dv += -prefac * (p.P^expfac * ker_i + q.P^expfac * ker_j) * x_pq


                # viscous term (kept from your original)
                ρp = max(p.rho, rho_floor)
                ρq = max(q.rho, rho_floor)
                #ker_visc = (q.m / ρq) * rDwendland2(0.5 * (p.h + q.h), r)
                #p.Dv += ρp * 8.0 * ker_visc * mu / (ρp * ρq) *
                dot(p.v - q.v, x_pq) / (r * r + 0.0025 * (p.h + q.h)^2) * x_pq
        end


end

# ==============
# Move & accelerate
# ==============

function move!(p::Particle)
        p.Dv = VEC0
        if p.type == FLUID
                p.x += dt * p.v
        end
end

function accelerate!(p::Particle)
        if p.type == FLUID
                p.v += 0.5 * dt * (p.Dv - g * VECY - damping_structure(p.x[2], zₜ, zᵦ, γᵣ) * VECY)
        end
end

# ==============
# Modified Verlet step (with pressure–entropy pieces)
# ==============

function verlet_step!(sys::ParticleSystem{Particle})
        # half-step acceleration & drift
        apply!(sys, accelerate!)
        apply!(sys, move!)
        create_cell_list!(sys)

	# compute density and smoothing length
	apply!(sys, compute_density!)
	apply!(sys, update_smoothing!)
	create_cell_list!(sys)
	
        # pressure–entropy: build P̄ from A
        apply!(sys, compute_pressure_bar!)
        apply!(sys, finalize_pressure_bar!)

        # thermodynamics from P̄ and ρ
        apply!(sys, find_temperature!)
        apply!(sys, find_pot_temp!)

        # forces
        apply!(sys, balance_of_momentum!)
        apply!(sys, accelerate!)
end

# ==============
# Diagnostics
# ==============

function avg_velocity(sys::ParticleSystem)::Float64
        v = 0.0
        for p in sys.particles
                v += norm(p.v)
        end
        v = v / length(sys.particles)
        return v
end

function max_velocity(sys::ParticleSystem)::Float64
        v = maximum(norm(p.v) for p in sys.particles)
        return v
end

# ==============
# Time loop
# ==============

function main()
        sys = make_system()
        out = new_pvd_file(folder_name)
        save_frame!(out, sys, export_vars...)

        nsteps = Int(round(t_end / dt))
        average_velocities = DataFrame(t=Float64[], u=Float64[])
        maximum_velocities = DataFrame(t=Float64[], u=Float64[])

        @show T0
        @show rho0
        @show mu
        @show c
        println("---------------------------")

        for k = 1:nsteps
                t = k * dt
                verlet_step!(sys)

                if (k % Int(round(dt_frame / dt)) == 0)
                        @show t
                        println("num. of particles = ", length(sys.particles))

                        u_avg = avg_velocity(sys)
                        @show u_avg
                        push!(average_velocities, (t, u_avg))

                        u_max = max_velocity(sys)
                        @show u_max
                        push!(maximum_velocities, (t, u_max))

                        save_frame!(out, sys, export_vars...)
                end
        end

        save_pvd_file(out)

        p1 = plot(
                average_velocities.t, average_velocities.u;
                xlabel="t (s)",
                ylabel="avg. velocity (m/s)",
                lc=:blue,
        )
        p2 = plot(
                maximum_velocities.t, maximum_velocities.u;
                xlabel="t (s)",
                ylabel="max. velocity (m/s)",
                lc=:orange,
        )
        plot(p1, p2; layout=(2, 1))
end

end # module


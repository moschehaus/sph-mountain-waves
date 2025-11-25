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

const folder_name = "../results/adiabatic_static_witch"
const export_vars = (:v, :rho, :P, :θ, :T, :type)

# ==============
# Constants
# ==============

# geometry parameters
const dom_height = 26e3        # height of the domain
const dom_length = 400e3       # length of the domain
const dr = dom_height / 75     # average particle distance
const bc_width = 6 * dr        # boundary width
const hₘ = 0.0                 # Witch of Agnesi: mountain height
const a = 0.0                  # Witch of Agnesi: mountain width

# smoothing paramaters
const η = 1.8 # prefactor for a average number of neighbours
const h0 = η * dr            # smoothing length


# physical parameters
const rho0 = 1.393             # reference density
const m0 = rho0 * dr * dr
const c = sqrt(65e3 * (7 / 5) / rho0) # speed of sound

# artifical parameters
const ν = 0.1 * h0 * c        # pressure stabilization
const ε = 0.01
const α = 0.1                # usually α = 0.05 - 0.2
const β = 2 * α               # usually β = 2 α 

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
const t_end = 10.0              # end of simulation
const dt_frame = t_end / 100   # frame interval

# particle types
const FLUID = 0.0
const WALL = 1.0
const MOUNTAIN = 2.0

# numerical safety floors
const rho_floor = 1e-6
const P_floor = 1e-10

include("../../examples/utils/new_packing.jl")

# ==============
# Particle type
# ==============

mutable struct Particle <: AbstractParticle
        h::Float64        # smoothing length
        x::RealVector     # position
        m::Float64        # mass
        v::RealVector     # velocity
        Dv::RealVector    # acceleration
	ρ̄
	ρ′
        rho::Float64      # density
        P::Float64        # kernel-averaged pressure (for output)
	P̄
	P′
        θ::Float64        # potential temperature
        T::Float64        # temperature
	T̄
	T′
        gGamma::RealVector
        type::Float64     # particle type
        A::Float64        # Hopkins entropy variable
	
        function Particle(x::RealVector, v::RealVector, type::Float64)
                obj = new(
                        h0,                # h, 
                        x, 0.0,            # x, m
                        v, VEC0,           # u, Dv
                        0.0,               # rho
                        0.0, 0.0,          # P, θ, 
                        0.0,               # T
                        VEC0,              # gGamma
                        type,              # type
                        0.0,               # A
                )

                # initial hydrostatic isothermal state 
                obj.T = T0
                obj.rho = rho0 * exp(-obj.x[2] * g / (R_mass * obj.T))
                obj.m = obj.rho * dr * dr#m0
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
        #grid = Grid(dr, :exp; K=g / (R_mass * T0))
        grid = Grid(dr, :hexagonal; K=1.0)
        domain = Rectangle(-dom_length / 2.0, 0.0, dom_length / 2.0, dom_height)
        fence = BoundaryLayer(domain, grid, bc_width)

        witch_profile(x) = (hₘ * a^2) / (x^2 + a^2)
        mountain = Specification(domain, x -> (x[2] <= witch_profile(x[1])))

        sys = ParticleSystem(Particle, domain + fence, h0)
        generate_particles!(sys, grid, domain - mountain, x -> Particle(x, VEC0, FLUID))
        generate_particles!(sys, grid, fence, x -> Particle(x, VEC0, WALL))
        generate_particles!(sys, grid, mountain, x -> Particle(x, VEC0, FLUID))

        create_cell_list!(sys)
        packing!(sys)
        create_cell_list!(sys)
        return sys
end

# ==============
# Thermodynamics helper: A and P
# ==============

@inbounds function reset_pressure_bar!(p::Particle)
        if p.type == FLUID
                p.P = 0.0
        end
end
@inbounds function compute_pressure_bar!(p::Particle, q::Particle, r::Float64)
        if p.type == FLUID
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
                p.rho += q.m * wendland2(p.h, r)
        end
end

@inbounds function reset_density!(p::Particle)
        if p.type == FLUID
                p.rho = 0.0
        end
end

@inbounds function update_smoothing!(p::Particle)
        if p.type == FLUID
                rho = max(p.rho, rho_floor)
                p.h = η * sqrt(p.m / rho)
        end
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
        v_pq = p.v - q.v
        dot_product = SmoothedParticles.dot(x_pq, v_pq)

        if p.type == FLUID
                prefac = q.m * (p.A * q.A)^(1 / γ)
                expfac = 1.0 - 2.0 / γ
                ker_i = rDwendland2(p.h, r)
                ker_j = rDwendland2(q.h, r)

                # pairwise conservative force
                pP = max(p.P, P_floor)
                qP = max(q.P, P_floor)
                p.Dv += -prefac * (pP^expfac * ker_i + qP^expfac * ker_j) * x_pq
                if dot_product < 0.0
                        h_ij = 0.5 * (p.h + q.h)
                        ker_ij = rDwendland2(h_ij, r)
                        prho = max(p.rho, rho_floor)
                        qrho = max(q.rho, rho_floor)
                        c_i = sqrt(γ * p.P / prho)
                        c_j = sqrt(γ * q.P / qrho)
                        c_ij = 0.5 * (c_i + c_j)
                        ρ_ij = 0.5 * (prho + qrho)
                        μ_ij = (h_ij * dot_product) / (r * r + ε * h_ij * h_ij)
                        π_ij = (-α * c_ij * μ_ij + β * μ_ij * μ_ij) / ρ_ij

                        # artificial viscous force
                        p.Dv += -q.m * π_ij * ker_ij * x_pq
                end
        end
end

# ==============
# Move & accelerate
# ==============

function move!(p::Particle)
        if p.type == FLUID
                p.x += dt * p.v
        end
end

function accelerate!(p::Particle)
        if p.type == FLUID
                p.v += 0.5 * dt * (p.Dv - g * VECY - damping_structure(p.x[2], zₜ, zᵦ, γᵣ) * VECY)
        end
        p.Dv = VEC0
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
        apply!(sys, reset_density!)
        apply!(sys, compute_density!)
        apply!(sys, update_smoothing!)
        create_cell_list!(sys)

        # pressure–entropy: build P̄ from A
        apply!(sys, reset_pressure_bar!)
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
                v += SmoothedParticles.norm(p.v)
        end
        v = v / length(sys.particles)
        return v
end

function max_velocity(sys::ParticleSystem)::Float64
        v = maximum(SmoothedParticles.norm(p.v) for p in sys.particles)
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


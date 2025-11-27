"""
 Static atmosphere above a mountain with the Witch of Agnesi profile:

 h(x) = (hₘ a²) / (x² + a²),

 all thermodynamic processes are adiabatic
"""
module PerturbedStaticWitch
export main

using Printf
using SmoothedParticles
using DataFrames
using Plots

unicodeplots()

const folder_name = "full_hopkins_perturbed_witch"
const export_vars = (:v, :ρ, :P, :θ, :T, :type)

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
const ρ0 = 1.393             # reference density
const m0 = ρ0 * dr * dr
const c = sqrt(65e3 * (7 / 5) / ρ0) # speed of sound

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
const T_bg = 250.0               # background (ie. initial) temperature

# temporal parameters
const dt = 0.01 * h0 / c       # time step
const t_end = 20.0              # end of simulation
const dt_frame = t_end / 100   # frame interval

# particle types
const FLUID = 0.0
const WALL = 1.0
const MOUNTAIN = 2.0

# numerical safety floors
const rho_floor = 1e-6
const P_floor = 1e-10


include(joinpath(UTILS_DIR,"new_packing.jl"))
# ==============
# Particle type
# ==============

mutable struct Particle <: AbstractParticle
        h::Float64        # smoothing length
        x::RealVector     # position
        m::Float64        # mass
        v::RealVector     # velocity
        Dv::RealVector    # acceleration
        ρ_bg::Float64     # background density
        ρ′::Float64       # density perturbation
        ρ::Float64        # total density
        P_bg::Float64     # background pressure
        P′::Float64       # pressure perturbation
        P::Float64        # total pressure
        θ_bg::Float64     # bakcground potential temperature
        θ′::Float64       # potential temperature perturbation
        θ::Float64        # total potential temperature
        T_bg::Float64     # background temperature
        T′::Float64       # temperature perturbation
        T::Float64        # total temperature
        type::Float64     # particle type
        A::Float64        # entropy-like variable
        A_bg::Float64        # entropy-like variable

        function Particle(x::RealVector, v::RealVector, type::Float64)
                obj = new(
                        h0,             # h 
                        x,              # x 	 	
                        0.0,            # m
                        v,              # v
                        VEC0,           # Dv
                        0.0,            # ρ_bg
                        0.0,            # ρ′
                        0.0,            # ρ
                        0.0,            # P_bg
                        0.0,            # P′
                        0.0,            # P
                        0.0,            # θ_bg 
                        0.0,            # θ′ 
                        0.0,            # θ 
                        0.0,            # T_bg
                        0.0,            # T′
                        0.0,            # T
                        type,           # type
                        0.0,            # A
                )

                # initial hydrostatic isothermal state 


                obj.T_bg = T_bg
                obj.ρ_bg = background_density(obj.x[2])
                obj.P_bg = background_pressure(obj.x[2])
                obj.θ_bg = background_pot_temperature(obj.x[2])
                obj.A_bg = background_entropy(obj.x[2])

                obj.ρ′ = 0.0
                obj.P′ = 0.0
                obj.T′ = 0.0
                obj.θ′ = 0.0

                obj.T = obj.T′ + T_bg
                obj.ρ = obj.ρ′ + obj.ρ_bg
                obj.P = obj.P′ + obj.P_bg
                obj.θ = obj.θ′ + obj.θ_bg

                obj.m = obj.ρ * dr^2
                obj.A = obj.P / obj.ρ^γ

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
        #packing!(sys)
        #create_cell_list!(sys)
        return sys
end

"""
Background (ie. initial) density, pressure and potential temperature distribution
"""


function background_density(y::Float64)
        return ρ0 * exp(-y * g / (R_mass * T_bg))
end

function background_pressure(y::Float64)
        ρ_bg = background_density(y)
        return R_mass * T_bg * ρ_bg
end

function background_pot_temperature(y::Float64)
        P_bg = background_pressure(y)
        return T_bg * (((T_bg * R_gas * ρ0) / P_bg))^(2 / 7)
end

function background_entropy(y::Float64)
        P_bg = background_pressure(y)
        ρ_bg = background_density(y)
        return P_bg / ρ_bg^γ
end

# ==============
# Pressure computation
# ==============

@inbounds function reset_pressure!(p::Particle)
        p.P = 0.0
        p.P′ = 0.0 #this should not be necessary; robustness precaution
end

@inbounds function compute_pressure!(p::Particle, q::Particle, r::Float64)
        ker = wendland2(0.5 * (p.h + q.h), r)
        p.P += q.m * q.A^(1 / γ) * ker
end

@inbounds function finalize_pressure!(p::Particle)
        p.P = p.P^γ
        p.P_bg = background_pressure(p.x[2])
        p.P′ = p.P - p.P_bg
end

# ==============
# Thermodynamics (simplified for P–A)
# ==============

@inbounds function find_temperature!(p::Particle)
        p.T = p.P / (R_mass * p.ρ)
        p.T′ = p.T - p.T_bg
end

@inbounds function find_pot_temp!(p::Particle)
        p.θ = p.T * (((T_bg * R_gas * ρ0) / p.P))^(2 / 7)
        p.θ_bg = background_pot_temperature(p.x[2])
        p.θ′ = p.θ - p.θ_bg
end

# ==============
# Smoothing-length & density evolution
# ==============

@inbounds function reset_density!(p::Particle)
        p.ρ = 0.0
        p.ρ′ = 0.0 # this should not be necessary; robustness precaution
end


@inbounds function compute_density!(p::Particle, q::Particle, r::Float64)
        p.ρ += q.m * wendland2(p.h, r)
end

@inbounds function finalize_density!(p::Particle)
        p.ρ_bg = background_density(p.x[2])
        p.ρ′ = p.ρ - p.ρ_bg
end

@inbounds function update_smoothing!(p::Particle)
        rho = max(p.ρ, rho_floor)
        p.h = η * sqrt(p.m / rho)
end


# ==============
# Rayleigh damping
# ==============

function damping_structure(z, zₜ, zᵦ, γᵣ)
        if z >= (zₜ - zᵦ)
                return -γᵣ * (sin(π / 2 * (1 - (zₜ - zᵦ) / zᵦ)))^2 * VECY
        else
                return VEC0
        end
end

function buyoancy_force(p::Particle)
        return -g * VECY * p.ρ′ / p.ρ # the (density) of gravity is - g * VECY

end
# ==============
# P–A momentum equation 
# ==============

@inbounds function balance_of_momentum!(p::Particle, q::Particle, r::Float64)
        x_pq = p.x - q.x
        v_pq = p.v - q.v
        dot_product = SmoothedParticles.dot(x_pq, v_pq)

        prefac = q.m * (p.A * q.A)^(1 / γ)
        expfac = 1.0 - 2.0 / γ
        ker_i = rDwendland2(p.h, r)
        ker_j = rDwendland2(q.h, r)
        pP = max(P_floor, p.P)
        qP = max(P_floor, q.P)

        # acceleration due the gradient of total pressure
        a_tot = -prefac * (pP^expfac * ker_i + qP^expfac * ker_j) * x_pq

        prefac_bg = q.m * (p.A_bg * q.A_bg)^(1 / γ)
        pP_bg = max(P_floor, p.P_bg)
        qP_bg = max(P_floor, q.P_bg)

        # acceleration due to the gradient of background pressure
        a_bg = -prefac_bg * (pP_bg^expfac * ker_i + qP_bg^expfac * ker_j) * x_pq

        # total acceleration
        p.Dv += a_tot - a_bg


        # artificial viscous force
        if dot_product < 0.0
                h_ij = 0.5 * (p.h + q.h)
                ker_ij = rDwendland2(h_ij, r)
                prho = max(p.ρ, rho_floor)
                qrho = max(q.ρ, rho_floor)
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
                p.v += 0.5 * dt * (p.Dv + buyoancy_force(p) + damping_structure(p.x[2], zₜ, zᵦ, γᵣ)) # this is a vector sum
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
        apply!(sys, finalize_density!)
        apply!(sys, update_smoothing!)
        create_cell_list!(sys)

        # pressure–entropy: build P̄ from A
        apply!(sys, reset_pressure!)
        apply!(sys, compute_pressure!)
        apply!(sys, finalize_pressure!)

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
        outpath = joinpath(RESULTS_DIR, folder_name)
        out = new_pvd_file(outpath)
        save_frame!(out, sys, export_vars...)

        nsteps = Int(round(t_end / dt))
        average_velocities = DataFrame(t=Float64[], u=Float64[])
        maximum_velocities = DataFrame(t=Float64[], u=Float64[])

        @show T_bg
        @show ρ0
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
	#savefig(joinpath(outpath,"velocities.pdf"))
end

end # module


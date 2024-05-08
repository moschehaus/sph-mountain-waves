#=

# 8: Adiabatic expansion 

```@raw html
	<img src='../assets/fixpa.png' alt='missing' width="50%" height="50%" /><br>
```
Simulation of adiabatic expansion of a viscous ideal gas.
=#

module adiabatic_mass

using Printf
using SmoothedParticles
using Parameters
using Plots
using DataFrames # to store the csv file
using CSV# to store the csv file
include("../../examples/utils/FixPA.jl")
include("../../examples/utils/entropy.jl")
include("../../examples/utils/ICR.jl")
using .FixPA
using .entropy
using LaTeXStrings #for better legends
using Random, Distributions
using LsqFit
using SparseArrays
using LinearAlgebra: I, det


#using ReadVTK  #not implemented
#using VTKDataIO

#=
Declare constant parameters
=#

##physical
const dr = 2.0e-2          #average particle distance (decrease to make finer simulation)
const h = 3.0*dr           #size of kernel support
const g = -9.8*VECY  #gravitational acceleration
const mu =  0.0#1.0e-03 #8.4e-4          #dynamic viscosity of water
const gamma = 1.4

const folder_name = "adiabatic_mass_mu_"*string(mu)
const cv = 1.0
const p0 = 10.0
const rho0 = 10.0
const c0 = sqrt(p0*gamma/rho0)
const m = rho0*dr^2        #particle mass
const kB = 1.380649E-23

const m0 = rho0*dr*dr
const S0 = m0*cv*log(p0/(gamma*rho0^gamma))
const T0 = gamma*rho0^(gamma-1)/(cv*(gamma-1))*exp(S0/(m0*cv))
const sigma = sqrt(kB*T0/m)
@show T0
@show rho0
@show m0
@show S0
@show mu
const epsilon = 1.0E-06
@show c0


##geometrical
const box_height = 1.0
const box_width = 1.0
const wall_width = 2.5*dr
const slit_height = box_height/10

##artificial
#const c = 50.0             #numerical speed of sound
const dr_wall = 0.95*dr
const E_wall = 10*norm(g)
const eps = 1e-6

##temporal
const dt = 0.001*h/c0
const t_end = 10.0 
const dt_frame = t_end/1000
@show t_end

##particle types
const FLUID = 0.
const WALL = 1.
const EMPTY = 2.


mutable struct Particle <: AbstractParticle
	x::RealVector
    m::Float64
    S::Float64 
    v::RealVector 
    a::RealVector 
    rho::Float64 
    s::Float64 
    P::Float64 
    T::Float64 
	type::Float64 #particle type
	Drho::Float64 #density increment
	Ds::Float64 #entropy density increment
	Particle(x::RealVector, type::Float64) = begin
		return new(x, m0, S0, VEC0, VEC0, 0.0, 0.0, 0.0, 0.0, type, 0.0, 0.0)
	end
end

#=
Define geometry and make particles
=#

function make_system()
	grid = Grid(dr, :square)
	boxL = Rectangle(0., 0., box_width-wall_width/2, box_width)
	boxR = Rectangle(box_width+wall_width/2, 0., 2*box_width, box_width)
	wallL = BoundaryLayer(boxL, grid, wall_width)
	wallR = BoundaryLayer(boxR, grid, wall_width)
	sys = ParticleSystem(Particle, boxL + wallL+wallR, h)

	#wallL = Specification(wallL, x -> x[2] <= wall_width)
	#wallR = Specification(wallR, x -> x[2] <= wall_width)

	generate_particles!(sys, grid, boxL, x -> Particle(x, FLUID))
	#generate_particles!(sys, grid, boxR, x -> Particle(x, FLUID))
	generate_particles!(sys, grid, wallL, x -> Particle(x, WALL))
	generate_particles!(sys, grid, wallR, x -> Particle(x, WALL))

	Random.seed!(42) #
	dist = MvNormal(2, sigma) #nondimensional distribution of velocities
	for i in 1:length(sys.particles)
		p = sys.particles[i]
		if p.type == WALL
			if (p.x[1] >= box_width - wall_width) && (p.x[1]<= box_width+wall_width) && (p.x[2] >= box_height/2 - slit_height) && (p.x[2]<= box_height/2 + slit_height)
				p.type = EMPTY
			end
		elseif p.type == FLUID
			vxy = rand(dist, 1)
			p.v = RealVector((vxy[1], vxy[2], 0.0))
		end
	end

	return sys
end

#=
Define particle interactions
=#

@inbounds function internal_force!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
		ker = q.m*rDwendland2(h,r)
    	x_pq = p.x - q.x
		#pressure
		p.a += -ker*(p.P/p.rho^2 + q.P/q.rho^2)*x_pq
		#entropic terms
		#p.a += -ker*(p.T/p.rho*(q.S/q.m - p.s/p.rho))*x_pq
  		#p.a += -ker*(q.T/q.rho*(p.S/p.m - q.s/q.rho))*x_pq

		#alternative entropic
    	#p.a += -ker*(p.T/p.rho*(q.S/q.m - p.S/p.m))*(p.x - q.x)
    	#p.a += -ker*(q.T/q.rho*(p.S/p.m - q.S/q.m))*(p.x - q.x)
		#ker = m*rDwendland2(h,r)
		#p.a += -ker*(p.P/rho0^2 + q.P/rho0^2)*(p.x - q.x)
    	p.a += 8.0*ker*mu/(p.rho*q.rho)*dot(p.v-q.v, x_pq)/(r*r + 0.01*h*h)*x_pq
		#p.a += 8*ker*mu/(p.rho*q.rho)*dot(p.v - q.v, p.x-q.x)*(p.x-q.x)/(dot(p.x-q.x,p.x-q.x)+epsilon)
		#p.a += +2*ker*mu/rho0^2*(p.v - q.v)
	elseif p.type == FLUID && q.type == WALL && r < dr_wall
		s2 = (dr_wall^2 + eps^2)/(r^2 + eps^2)
		p.a += -E_wall/(r^2 + eps^2)*(s2 - s2^2)*(p.x - q.x)
	end	
end

function reset_a!(p::Particle)
    p.a = zero(RealVector)
end

function reset_rho!(p::Particle)
    p.rho = 0.0
end

function move!(p::Particle)
	if p.type == FLUID
		p.a = VEC0
		p.x += dt*p.v
		#reset rho, s and a
		p.rho = 0.
		#p.s = 0.
		p.a = VEC0
	end
end

function accelerate!(p::Particle)
	if p.type == FLUID
		#p.v = rev_add(p.v, 0.5*dt*p.a)
		p.v = p.v + 0.5*dt*p.a
	end
end

@inbounds function find_rho!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
        p.rho += q.m*wendland2(h,r)
        #p.s   += q.S*wendland2(h,r)
    end
end

@inbounds function find_s!(p::Particle)
	if p.type == FLUID
        p.s   = p.S*p.rho/p.m
    end
end



#@inbounds function entropy_production!(p::Particle, q::Particle, r::Float64)
#	if p.type == FLUID && q.type == FLUID
#		p.S += -0.5*8*mu*p.m*q.m/(p.rho*q.rho)*rDwendland2(h,r)*(dot(p.v-q.v, p.x-q.x))^2/(dot(p.x-q.x,p.x-q.x)+epsilon)/p.T
#		#p.S += -mu/rho0^2 *m/(p.T+epsilon)*rDwendland2(h,r)*dot(p.v-q.v,p.v-q.v)
#    end
#end


@inbounds function find_rho0!(p::Particle, q::Particle, r::Float64)
    if p.type == FLUID && q.type == FLUID
		p.rho0 += q.m*wendland2(h,r)
        p.s   += q.S*wendland2(h,r)
	end
end

function find_P!(p::Particle)
	if p.type == FLUID
    	p.T = (p.rho^(gamma - 1.0))*exp(p.s/(p.rho*cv))/(cv*(gamma-1.0))
    	#p.P = (p.rho^gamma)*exp(p.s/(p.rho*cv))
		#p.P = ((p.rho^gamma - p.rho^(gamma-1.0)/(gamma-1.0)*p.s/cv)*exp(p.s/(p.rho*cv)) + p.rho*p.S/p.m*p.T) 
		p.P = (gamma-1.0)*p.rho*p.T*cv
	end
end

function LJ_potential(p::Particle, q::Particle, r::Float64)::Float64
	if q.type == WALL && p.type == FLUID && r < dr_wall
		s2 = (dr_wall^2 + eps^2)/(r^2 + eps^2)
		return m*E_wall*(0.25*s2^2 - 0.5*s2 + 0.25)
	else
		return 0.0
	end
end

function update_rho_and_s!(p::Particle)
	if p.type == FLUID
		#p.rho += dt*p.Drho 	
		#p.Drho = 0.0

		p.s += dt*p.Ds
		p.Ds = 0.0

		p.S = p.m*p.s/p.rho #regularize S_alpha
	end
end

function find_Drho_and_Ds!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
		ker = rDwendland2(h,r)
		x_pq = p.x - q.x
		v_pq = p.v - q.v
		#p.Drho +=  2.0*alpha*m*ker*(p.P - q.P)*p.rho/q.rho
		p.Ds += q.S * ker * dot(x_pq, v_pq)
	end
end

function entropy_production!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
		ker = rDwendland2(h,r)
		x_pq = p.x - q.x
		v_pq = p.v - q.v
    	p.S += - 4.0*p.m*q.m*ker*mu/(p.T*p.rho*q.rho)*dot(v_pq, x_pq)^2/(r*r + 0.01*h*h)*dt #viscous
	end
end


function energy_kinetic(sys::ParticleSystem)::Float64
	return sum(p -> 0.5*m*dot(p.v, p.v), sys.particles)
end

function left(sys::ParticleSystem)::Float64
	left_number = 0
	for p in sys.particles
		if p.x[1] <= box_width
			left_number += 1
		end
	end
	return left_number
end

function energy(sys::ParticleSystem)
	(E_kin, E_int, E_gra, E_wal, E_tot, S) = (0., 0., 0., 0., 0., 0.)
	for p in sys.particles
		if p.type == FLUID
			E_kin += 0.5*m*dot(p.v, p.v)
			E_int += p.m*cv* p.T
			#E_int +=  0.5*m*c^2*(p.rho - p.rho0)^2/rho0^2
			#E_gra += -m*dot(g, p.x)
			E_wal += SmoothedParticles.sum(sys, LJ_potential, p)
			S += p.S
		end
	end
	#E_tot = E_kin + E_int + E_gra + E_wal
	E_tot = E_kin +E_int + E_wal
	return (E_tot, E_kin, E_int, E_wal, S)
end

#=
Put everything into a time loop
=#
#
function verlet_step!(sys::ParticleSystem)
    apply!(sys, accelerate!)
    apply!(sys, move!)
    create_cell_list!(sys)
    #apply!(sys, reset_rho!)
    #apply!(sys, find_rho!, self = true)
    #apply!(sys, find_pressure!)
    apply!(sys, reset_a!)

    apply!(sys, find_rho!, self=true)
    apply!(sys, find_s!)

    #apply!(sys, find_Drho_and_Ds!, self=true)
    #apply!(sys, update_rho_and_s!)
    apply!(sys, find_P!)
    apply!(sys, entropy_production!)

    apply!(sys, internal_force!)
    apply!(sys, accelerate!)
end

function save_results!(out::SmoothedParticles.DataStorage, sys::ParticleSystem, k::Int64, E0::Float64)
    if (k %  Int64(round(dt_frame/dt)) == 0)
         save_frame!(out, sys, :v, :a, :type, :P, :s, :T, :rho, :S)
    end
end

function main() 
	sys = make_system()
	out = new_pvd_file(folder_name)
    #initialization
    create_cell_list!(sys)
    #apply!(sys, find_rho0!, self = true)
    apply!(sys, find_rho!, self = true)
    apply!(sys, find_s!)
    apply!(sys, find_P!)
    apply!(sys, internal_force!)

	N_of_particles = length(sys.particles)
	@show(N_of_particles)
	@show(m)

	step_final = Int64(round(t_end/dt))
	times = Float64[] #time instants
	#thermalize!(sys)
	E0 = energy(sys)[1]
	initial_T = average_T(sys)
	@show initial_T
	ls = Float64[]
	Ts = Float64[] # Entropy values
	Ekin = Float64[] # Kinetic energy values
	#Eg = Float64[] # Gravitational energy values
	Ewall = Float64[] # Wall energy values
	Eint = Float64[] # Internal energy values
	Etot = Float64[] # Internal energy values
	Ss = Float64[] # Internal energy values

	for k = 0 : step_final
        verlet_step!(sys)
        save_results!(out, sys, k, E0)
    	if k % round(step_final/100) == 0 # store a number of entropy values
       		@printf("t = %.6e\n", k*dt)
			#distr = velocity_histogram(sys, N = 100)
			#S = entropy_2D_MB(distr)
			#push!(Ss, S)
			#@show(S)

			push!(times, k*dt)
			left_number = left(sys)
			push!(ls, left_number)
			@show left_number

			#energy
			(E_tot, E_kin, E_int, E_wal, S) = energy(sys)
			@show E_tot
			push!(Etot, E_tot)
			@show E_kin
			push!(Ekin, E_kin)
			@show E_int
			push!(Eint, E_int)
			@show E_wal	
			push!(Ewall, E_wal)
			E_err = E_tot - E0
			@show E_err	
			T = average_T(sys)
			push!(Ts, T)
			@show T
			push!(Ss, S)
			@show S
			println("# of part. = ", length(sys.particles))
			println()
		end
	end

	# Plotting the energies in time
	p = plot(times, Etot, label = "E_tot",legend=:bottomright)
	savefig(p, folder_name*"/Etot.pdf")
	p = plot(times, Ekin, label = "E_kin",legend=:bottomright)
	savefig(p, folder_name*"/Ekin.pdf")
	p = plot(times, Eint, label = "E_int",legend=:bottomright)
	savefig(p, folder_name*"/Eint.pdf")
	p = plot(times, Ewall, label = "E_wall",legend=:bottomright)
	savefig(p, folder_name*"/Ewall.pdf")
	p = plot(times, Ts, label = "T",legend=:bottomright)
	savefig(p, folder_name*"/T.pdf")
	p = plot(times, Ss, label = "S",legend=:bottomright)
	savefig(p, folder_name*"/S.pdf")
	p = plot(times, ls, label = "left_number",legend=:bottomright)
	savefig(p, folder_name*"/left_number.pdf")

	df = DataFrame(time_steps = times, left = ls, E_total = Etot, E_kinetic = Ekin, E_internal = Eint, E_walls = Ewall, temperature = Ts, entropy = Ss)
	CSV.write(folder_name*"/results.csv", df)

	final_T = average_T(sys)
	@show initial_T
	@show final_T

	save_pvd_file(out)

end ## function main

function plot_left(left_file::String; fit=false)
    df = DataFrame(CSV.File(left_file))
    times = df[:, "time_steps"]
    ls = df[:, "left"]
	if fit
		model(t, p) = p[1] * exp.(-p[2] * t) .+ p[3]
		p0 = [ls[1]/2, 10.0, ls[1]/2]
		fitting = curve_fit(model, times, ls, p0)
		param = fitting.param
		@show param
		curve = param[1] * exp.(-param[2] .* times) .+ param[3]
		plot()
    	p = plot!(times, [ls, curve], legend=:topright, labels=["number of left particles"  "exponential fit"], linewidth=5, thickness_scaling=1)
	else
		plot()
    	p = plot!(times, ls, legend=:topright, label="number of left particles", linewidth=5, thickness_scaling=1)
	end
	savefig(p, folder_name*"/left.pdf")
end


function plot_energy(energy_file::String)
    df = DataFrame(CSV.File(energy_file))
    times = df[:, "time_steps"]
    e_pot = df[:, "E_graviational"]
	Delta_e_pot = e_pot[1]-e_pot[end]
	print("Delta e pot = ", Delta_e_pot)
    e_tot = df[:, "E_total"]
	e_tot0 = e_tot[1]
	e_tot = (e_tot .- e_tot0)./Delta_e_pot
    p = plot(times, e_tot, legend=:topright, label=L"\frac{E_{tot}-E_{tot}(0)}{E_g(end)-E_g(0)}")
	savefig(p, "./energy_tot_scaled.pdf")
end

function average_T(sys::ParticleSystem)::Float64
    T = 0.0
    n = 0
    for p in sys.particles
        if p.type == FLUID
            T += p.T
            n += 1
        end
    end
    return T/n
end

function Wab(p::Particle, q::Particle, r::Float64)::Float64
	return wendland2(h,r)
end

function W_inverse(sys::ParticleSystem)#::SparseMatrixCSC{Float64, Int64}
	n = length(sys.particles)
	W = SmoothedParticles.assemble_matrix(sys, Wab)
	print("Matrix aseembled\n")
	determinant = det(W)
	@show determinant
end

end ## module


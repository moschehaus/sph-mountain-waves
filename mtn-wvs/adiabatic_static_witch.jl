#=

# Flow around a mountain with the Witch of Agnesi profile:

h(x)=(hₘa²)/(x²+a²)

=#

module adiabatic_static_witch

using Printf
using SmoothedParticles
using Parameters
using Plots
using DataFrames # to store the csv file
using CSV# to store the csv file
include("../examples/utils/FixPA.jl")
include("../examples/utils/entropy.jl")
include("../examples/utils/ICR.jl")
using .FixPA
using .entropy
using LaTeXStrings #for better legends
using Random, Distributions
using LsqFit
using SparseArrays
using LinearAlgebra: I, det

const folder_name = "mtn-wvs/results/adiabatic_static_witch"
#using ReadVTK  #not implemented
#using VTKDataIO

#=
Declare constant parameters
=#


##geometrical
const dom_height = 26e3/1e5   #height of the domain 
const dom_length = 400e3/1e5  #length of the domain
const hₘ = 100.            #parameters for the Witch of Agnesi profile; mountain height
const a = 10e3           #parameters for the Witch of Agnesi profile; mountain width


##physical parameters
const dr = 2.0e-2          #average particle distance (decrease to make finer simulation)
const h = 3.0*dr           #size of kernel support# larger is better
const bc_width = h    #width of boundary layer                 

const g = -9.8*VECY  #gravitational acceleration
const mu = 15.98e-6		#dynamic viscosity
const U_max = 20.0       #maximum inflow velocity
const rho0 =1.177		 #referential fluid density


##thermodynamics
const gamma = 1.4
const cv = 1.0
const p0 = 10.0
const c0 = sqrt(p0*gamma/rho0)
const nu = 0.1*h*c0      #pressure stabilization
const kB = 1.380649E-23
const R_gas=287.05
const T=250

##meteorological parameters
const N=0.0196
const γᵣ=10*N
const zᵦ=16e3
const zₜ=dom_height

##initial parameters
const m0 = rho0*dr*dr
const m = m0
#const S0 = m0*cv*log(p0/(gamma*rho0^gamma))
const T0 = 250 #gamma*rho0^(gamma-1)/(cv*(gamma-1))*exp(S0/(m0*cv))
const sigma = sqrt(kB*T0/m)
@show T0
@show rho0
@show m0
#@show S0
@show mu
const epsilon = 1.0E-06
@show c0

##artificial
#const c = 50.0             #numerical speed of sound
const dr_wall = 0.95*dr
const E_wall = 10*norm(g)
const eps = 1e-6

##temporal
const dt = 0.1*h/c0
const t_end = 1.0
const dt_frame = t_end/200
@show t_end

##particle types
const FLUID = 0.
const WALL = 1.
const EMPTY = 2.
const INFLOW = 3.
const OBSTACLE = 4.

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

	function Particle(x,type)
        obj=new(x, m0, 0.0, VEC0, VEC0, 0.0, 0.0, 0.0, T0, type, 0.0, 0.0)
        obj.rho=rho0*exp(-obj.x[2]*(-g[2])/(R_gas*obj.T))
		obj.m=obj.rho*dr^2
		obj.P=(gamma-1.0)*obj.rho*obj.T*cv
		obj.S=obj.m*cv*log(obj.P/(gamma*obj.rho^gamma))
        return obj
    end
end

#=
Define geometry and make particles
=#

function make_system()
	grid = Grid(dr, :square)
	domain=Rectangle(-dom_length/2,0.,dom_length/2,dom_height)
	
	fence=BoundaryLayer(domain,grid,bc_width)
	#ground = Specification(fence,x->(x[2] < 0 && x[1]<=dom_length/2))
    #sky=Specification(fence,x->(x[2]>dom_height && x[1]<=dom_length/2))
    #wind=Specification(fence,x->((x[1]<=-dom_length/2) && (x[2]>=0 && x[2]<=dom_height)))
    #mountain=Witch(hₘ,a)

	sys = ParticleSystem(Particle, domain+fence, h)
	generate_particles!(sys, grid, domain, x -> Particle(x, FLUID))
	generate_particles!(sys, grid, fence, x -> Particle(x, WALL))
	#generate_particles!(sys, grid, mountain, x -> Particle(x, OBSTACLE))
	#generate_particles!(sys, grid, wind, x -> Particle(x, INFLOW))

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

		p.a += 8.0*ker*mu/(p.rho*q.rho)*dot(p.v-q.v, x_pq)/(r*r + 0.01*h*h)*x_pq
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
		p.v = p.v + 0.5*dt*(p.a+g)
	end
end

function set_density!(p::Particle)
    if p.type==FLUID 
        p.rho=rho0*exp(-p.x[2]*(-g[2])	/(R_gas*T))
    else
        p.rho=rho0    
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
	Eg = Float64[] # Gravitational energy values
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

	save_pvd_file(out)

end ## function main

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

#=

# Static atmosphere above a mountain with the Witch of Agnesi profile:

h(x)=(hₘa²)/(x²+a²),

all thermodynamic proccesses are adiabatic

=#

module adiabatic_static_witch

using Printf
using SmoothedParticles
include("../../examples/utils/atmo_packing.jl")
using .atmo_packing

const folder_name = "../results/adiabatic_static_witch"
const export_vars = (:u, :rho, :P, :θ, :T, :type)

#=
Declare constants
=#

#geometry parameters
const dom_height = 26e3   #height of the domain 
const dom_length = 400e3  #length of the domain
const dr = dom_height/50  #average particle distance (decrease to make finer simulation)
const h = 1.8*dr          #smoothing length    
const bc_width = 6*dr     #boundary width
const hₘ = 100            #parameters for the Witch of Agnesi profile; mountain height
const a = 10e3            #parameters for the Witch of Agnesi profile; mountain width

#physical parameters
const rho0 =1.393		 #referential fluid density
const mu =15.98e-6		 #dynamic viscosity
const c = sqrt(65e3*(7/5)/rho0)	 #speed of sound

#meteorological parameters
const N=sqrt(0.0196)     #Brunt-Vaisala frequency
const g=9.81             #gravity
const R_mass=287.05      #specific molar gas constant
const γᵣ=10*N            #damping coefficient
const zᵦ=12e3             #bottom part of the damping layer
const zₜ=dom_height      #top part of the damping layer

#thermodynamical parameters
const R_gas=8.314        #universal molar gas constant
const cp=7*R_mass/2      #specific molar heat capacity at a constant pressure
const cv=cp-R_mass		 #specific molar heat capacity at a constant volume
const γ=cp/cv			 #poisson constant
const T0=250			 #initial temperature

#temporal parameters
const dt = 0.01*h/c   #time step
const t_end =100 #end of simulation
const dt_frame =t_end/100 #how often data is saved

#particle types
const FLUID = 0.0
const WALL = 1.0
const MOUNTAIN = 2.0

mutable struct Particle <: AbstractParticle
	x::RealVector  #position
	m::Float64     #mass
	u::RealVector  #velocity
	Du::RealVector #acceleration
	rho::Float64   #density
	P::Float64     #pressure
	θ::Float64     #potential temperature
	S::Float64     #entropy
	s::Float64 	   #entropy density
	T::Float64 	   #temperature 
	gGamma::RealVector #for the packing algorithm, "uneveness" of particle distribution
	type::Float64  #particle type

	function Particle(x::RealVector,u::RealVector,type::Float64)
		obj=new(x,0.0,u,VEC0,0.0,0.0,0.0,0.0,0.0,0.0,VEC0,type)
		obj.T=T0 #prescribe initial temperature
		obj.rho=rho0*exp(-obj.x[2]*g/(R_mass*obj.T))
		obj.m=obj.rho*dr^2
		obj.P=R_mass*obj.T*obj.rho
		obj.θ=obj.T*(((T0*R_gas*rho0)/obj.P)^(2))^(1/7) #2/7 is exactly R_gas/cp
		obj.S=obj.m *cv*log((cv*obj.T*(γ - 1))/(γ * obj.rho^(γ - 1))) #calculate initial entropy from the initial temperature
		return obj
	end
end

#=
Define geometry and make particles
=#

function make_system()
	grid=Grid(dr,:hexagonal)
	domain = Rectangle(-dom_length/2.0, 0.0, dom_length/2.0,  dom_height)
	fence = BoundaryLayer(domain,grid,bc_width)

	witch_profile(x)=(hₘ*a^2)/(x^2+a^2)
	mountain=Specification(domain,x->(x[2]<=witch_profile(x[1])))

	sys = ParticleSystem(Particle,domain+fence, h)
	generate_particles!(sys,grid,domain-mountain,x -> Particle(x,VEC0,FLUID))
	generate_particles!(sys,grid,fence,x -> Particle(x,VEC0,WALL))
	generate_particles!(sys,grid,mountain,x -> Particle(x,VEC0,MOUNTAIN))

	create_cell_list!(sys)
	improved_sys=atmo_packing.packing(sys,1e-10,1e-10,100) #packing algorithm]
	create_cell_list!(improved_sys)
	apply!(improved_sys, find_s!)
	apply!(improved_sys,internal_force!)
	return improved_sys
end

#=
Define particle interactions 	
=#

@inbounds function internal_force!(p::Particle, q::Particle, r::Float64)
	ker = q.m*rDwendland2(h,r)
	x_pq = p.x - q.x

	#pressure terms
	p.Du += -ker*(p.P/p.rho^2 + q.P/q.rho^2)*x_pq
	p.Du += 8.0*ker*mu/(p.rho*q.rho)*dot(p.u-q.u, x_pq)/(r*r + 0.01*h*h)*x_pq
end

#= 
Calculate density, pressure, temperature, potential temperature, entropy, entropy density
=#

@inbounds function find_density!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
		p.rho += q.m*wendland2(h,r)
	end
end

@inbounds function find_s!(p::Particle)
	if p.type == FLUID
		p.s   = p.S*p.rho/p.m
	end
end

function find_pressure!(p::Particle)
	if p.type == FLUID
		p.T = (p.rho^(γ-1.0))*exp(p.s/(p.rho*cv))/(cv*(γ-1.0))
		p.P = R_mass*p.rho*p.T
	end
end

function find_pot_temp!(p::Particle)
	if p.type==FLUID
		p.θ=p.T*(((T0*R_gas*rho0)/p.P)^(2))^(1/7) #2/7 is exactly R_gas/cp
	end
end

function entropy_production!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
		ker = rDwendland2(h,r)
		x_pq = p.x - q.x
		u_pq = p.u - q.u
		p.S += - 4.0*p.m*q.m*ker*mu/(p.T*p.rho*q.rho)*dot(u_pq, x_pq)^2/(r*r + 0.01*h*h)*dt #viscous
	end
end

#=
### Rayleigh damping
=#

function damping_structure(z,zₜ,zᵦ,γᵣ)
	if z >= (zₜ-zᵦ)
		return γᵣ*(sin(π/2*(1-(zₜ-zᵦ)/zᵦ)))^2
	else    
		return 0
	end
end               

#=
### Move and accelerate
=#

function move!(p::Particle)
	p.Du = VEC0
	if p.type == FLUID
		p.x += dt*p.u
		p.rho=0.0     #we want to reset the density only for the fluid, so we call it here
	end
end

function accelerate!(p::Particle)
	if p.type == FLUID
		p.u += 0.5*dt*(p.Du - g*VECY-damping_structure(p.x[2],zₜ,zᵦ,γᵣ)*VECY)
	end
end

#=
Modifed Verlet scheme
=#

function verlet_step!(sys::ParticleSystem)
	apply!(sys, accelerate!)
	apply!(sys, move!)
	create_cell_list!(sys)

	apply!(sys, find_density!, self=true)
	apply!(sys, find_s!)
	apply!(sys, find_pressure!)
	apply!(sys, entropy_production!)
	apply!(sys, internal_force!)
	apply!(sys, accelerate!)
end

#=
Put everything into a time loop
=#

function  main()
	sys = make_system()
	out = new_pvd_file(folder_name)
	save_frame!(out, sys, export_vars...)
	nsteps = Int64(round(t_end/dt))
	@show T0
	@show rho0
	@show mu
	@show c

	#a modified Verlet scheme
	for k = 1 : nsteps 
		t = k*dt
		verlet_step!(sys)
		#save data at selected 
		if (k %  Int64(round(dt_frame/dt)) == 0)
			@show t
			println("num. of particles = ", length(sys.particles))
			save_frame!(out, sys, export_vars...)
		end
	end
	save_pvd_file(out)
end

end


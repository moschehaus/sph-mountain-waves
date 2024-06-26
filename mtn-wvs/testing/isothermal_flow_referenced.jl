#=

# Isothermal flow around a mountain with the Witch of Agnesi profile. The outflow is at the top to counter decompression.
Remember that since the simulation starts from a fixed initial_state.vtp, the parameters have to match with reference_state.jl

h(x)=(hₘa²)/(x²+a²)

=#

module isothermal_flow_referenced

using Printf    
using Parameters
using SmoothedParticles
include("../../examples/utils/atmo_packing.jl")
using .atmo_packing

const folder_name = "../results/isothermal_flow_referenced"
const export_vars = (:u, :P, :θ, :rho, :type)

#=
Declare constants
=#

#geometry parameters
const dom_height = 26e3   #height of the domain
const dom_length = 200e3  #length of the domain
const a = 10e3            #parameters for the Witch of Agnesi profile; mountain width
const hₘ = 8e3            #parameters for the Witch of Agnesi profile; mountain height
const dr = dom_height/150 #average particle distance (decrease to make finer simulation)
const h = 1.8*dr          #smoothing length
const bc_width = 6*dr     #boundary width

#physical parameters
const U_max = 20.0       #maximum inflow velocity
const rho0 =1.393		 #referential fluid density
const mu = 15.98e-6		 #dynamic viscosity
const c = sqrt(65e3*(7/5)/rho0)	 #speed of sound

#meteorological parameters
const N=sqrt(0.0196)     #Brunt-Vaisala frequency
const g=9.81             #gravity
const R_mass=287.05      #specific molar gas constant
const R_gas=8.314        #universal molar gas constant
const cp=7*R_gas/2       #molar heat capacity at a constant pressure
const T=250              #constant temperature
const γᵣ=10*N            #damping coefficient
const zᵦ=12e3            #bottom part of the damping layer
const zₜ=dom_height      #top part of the damping layer

#temporal parameters
const dt = 0.1*h/c   #time step
const t_end =20 #end of simulation
const dt_frame =t_end/20 #how often data is saved

#particle types
const FLUID = 0.0
const WALL = 1.0
const MOUNTAIN = 2.0
const INFLOW = 3.0
const OUTFLOW = 4.0
#do not change the numbers! fluid, wall and mountain has to be the same as in reference_state.jl

#=
Declare variables to be stored in a Particle
=#

mutable struct Particle <: AbstractParticle
    x::RealVector  #position
    u::RealVector  #velocity
    Du::RealVector #acceleration
    rho::Float64   #density
    Drho::Float64  #density rate
    m::Float64     #mass
    P::Float64     #pressure
    θ::Float64     #potential temperature
    type::Float64  #particle type
    gGamma::RealVector #for the packing algorithm, "uneveness" of particle distribution
    
    function Particle(x::RealVector, u::RealVector, type::Float64)
        obj = new(x, u, VEC0,0.0,0.0,0.0,0.0,0.0,type,VEC0)  
        obj.rho=rho0*exp(-obj.x[2]*g/(R_mass*T))
        obj.m=obj.rho*dr^2 # set the mass of the particle according to its density
        obj.P=obj.rho*T*R_mass - (rho0*exp(-dom_height*g/(R_mass*T))*T*R_mass) # !by subtracting the pressure at the top, at the top there is p=0
        obj.θ=T*(((T*R_gas*rho0)/obj.P)^(2))^(1/7) #2/7 is exactly R_gas/cp
        return obj
    end
end

#=
### Define geometry and create particles
=#

function make_system()
    grid=Grid(dr,:hexagonal)
    domain = Rectangle(-dom_length/3.0, 0.0, 2*dom_length/3.0,  dom_height)
    fence=BoundaryLayer(domain,grid,bc_width)
    sys=ParticleSystem(Particle,domain+fence,h)
    import_particles!(sys,"initial_state.vtp",x->Particle(x,VEC0,FLUID)) #it should not matter what the type is set to
    apply!(sys,set_particle_type!) #sets the correct particle types
    filter!(p->(p.type != OUTFLOW),sys.particles) #removes particles at the outflow region, which is up!
    apply!(sys,initialize_system!) #sets the correct initial velocities 
    create_cell_list!(sys)

    apply!(sys,set_density!) #sets ρ,p,θ according to the actual positions 
    apply!(sys,find_pressure!)
    apply!(sys,find_pot_temp!)

	return sys
end

#=
### Initialize the system after packing
=#

function set_particle_type!(p::Particle)
    if (p.x[1]>(2*dom_length/3.0-dom_height) && p.x[1]<2*dom_length/3.0 && p.x[2]>dom_height) #makes "an hole in the roof"
        p.type=OUTFLOW
    elseif (p.x[1]<-dom_length/3.0 && p.x[2]>0 && p.x[2]<dom_height)
        p.type=INFLOW
    end    
end

function initialize_system!(p::Particle)
    if (p.type==FLUID || p.type==INFLOW)
        p.u=U_max*VECX
    elseif (p.type==MOUNTAIN || p.type==WALL)
        p.u=VEC0
    end
end

#=
### Deploy SPH equations
=#

@inbounds function balance_of_mass!(p::Particle, q::Particle, r::Float64)
	ker =q.m*rDwendland2(h,r)
	p.Drho += ker*(dot(p.x-q.x, p.u-q.u))
end

@inbounds function internal_force!(p::Particle, q::Particle, r::Float64)
	ker = q.m*rDwendland2(h,r)
    x_pq = p.x - q.x
	p.Du += -ker*(p.P/p.rho^2 + q.P/q.rho^2)*x_pq
    p.Du += 8.0*ker*mu/(p.rho*q.rho)*dot(p.u - q.u, x_pq)/(r*r + 0.01*h*h)*x_pq
end

#=
### Calculate pressure, density, potential temperature
=#

function find_pressure!(p::Particle)
    p.rho+=p.Drho*dt
	p.Drho = 0.0
	p.P = p.rho*R_mass*T   
end

function set_density!(p::Particle)
    p.rho=rho0*exp(-p.x[2]*g/(R_mass*T))
end


function find_pot_temp!(p::Particle)
    p.θ=T*(((T*R_gas*rho0)/p.P)^(2))^(1/7) #2/7 is exactly R_gas/cp
end

#=
### Generate new particles
=#

function add_new_particles!(sys::ParticleSystem)
    new_particles = Particle[]
    for p in sys.particles
        if (p.type == INFLOW && p.x[1] >= -dom_length/3.0)
            p.type = FLUID
            x = p.x - bc_width*VECX
            newp = Particle(x,U_max*VECX,INFLOW)
            push!(new_particles, newp)
        end
    end
    append!(sys.particles, new_particles)
end

#=
### Rayleigh damping
=#

function damping_structure(z,zₜ,zᵦ,γᵣ)
    if z >= (zₜ-zᵦ)
        return γᵣ*(sin(π/2*(1-(zₜ-zᵦ)/zᵦ)))^2
    else    
        return 0.0
    end
end            

#=
### Move and accelerate
=#

function move!(p::Particle)
	p.Du = VEC0
	if p.type == FLUID || p.type == INFLOW
		p.x += dt*p.u
	end
end

function accelerate!(p::Particle)
	if p.type == FLUID 
		p.u += 0.5*dt*(p.Du - g*VECY-damping_structure(p.x[2],zₜ,zᵦ,γᵣ)*VECY)
	end
end

#=
### Modified Verlet scheme
=#

function verlet_step!(sys::ParticleSystem)
    apply!(sys, accelerate!)
    apply!(sys, move!)
    add_new_particles!(sys)
    create_cell_list!(sys)

    apply!(sys, balance_of_mass!)
    apply!(sys, find_pressure!)
    apply!(sys,find_pot_temp!)
    apply!(sys, internal_force!)
    apply!(sys, accelerate!)
end

#=
### Put everything in a time loop
=#

function  main()
    sys = make_system()
	out = new_pvd_file(folder_name)
    save_frame!(out, sys, export_vars...)
    nsteps = Int64(round(t_end/dt))
    @show T
    @show U_max
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



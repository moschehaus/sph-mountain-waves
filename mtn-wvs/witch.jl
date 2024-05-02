#=

# Flow around a mountain with the Witch of Agnesi profile:

h(x)=(hₘa²)/(x²+a²)

=#

module witch

using Printf    
using Parameters
using SmoothedParticles

const folder_name = "mtn-wvs/results/witch"
const export_vars = (:u, :P, :rho, :type)

#=
Declare constants
=#

#geometry parameters
#scale=1e0                       #scale to make simulation smaller for testing
const dom_height = 26e3   #height of the domain 
const dom_length = 400e3  #length of the domain
const dr = dom_height/50	    #average particle distance (decrease to make finer simulation)
const h = 2.4*dr              
const bc_width = 12*dr                 
const hₘ = 100.            #parameters for the Witch of Agnesi profile; mountain height
const a = 10e3           #parameters for the Witch of Agnesi profile; mountain width



#physical parameters
const U_max = 20.0       #maximum inflow velocity
const rho0 =1.177		 #referential fluid density
const m0 = rho0*dr^2	 #particle mass
const c = 10.0*U_max	 #numerical speed of sound
const mu = 15.98e-6		#dynamic viscosity
const nu = 0.1*h*c      #pressure stabilization

#meteorological parameters
const N=0.0196
const g=9.81
const R_gas=287.05
const T=250
const γᵣ=10*N
const zᵦ=16e3
const zₜ=dom_height



#temporal parameters
const dt = 0.1*h/c                     #time step
const t_end = 200.0              #end of simulation, 4*3600
const dt_frame = max(dt, t_end/200)    #how often data is saved

#particle types
const FLUID = 0.0
const INFLOW = 1.0
const WALL = 2.0

#=
Declare variables to be stored in a Particle
=#

mutable struct Particle <: AbstractParticle
    x::RealVector # position
    u::RealVector # velocity
    Du::RealVector # acceleration
    rho::Float64 # density
    Drho::Float64  # density rate
    m::Float64 #mass of the particle
    P::Float64  # pressure
    type::Float64 # particle type
    
    function Particle(x::RealVector, u::RealVector, type::Float64)
        obj = new(x, u, VEC0,0.0,0.0,0.0, 0.0, type)  # initialize Drho, P, Du as zero
        set_density!(obj)  # call set_density! function to set rho
        obj.m=obj.rho*dr^2 # set the mass of the particle according to its density
        return obj
    end
end


#=
### Define geometry and create particles
=#

function make_system()
    grid=Grid(dr,:square)
    domain = Rectangle(-dom_length/2.0, 0.0, dom_length/2, dom_height)
    fence = BoundaryLayer(domain,grid,bc_width)
    #ground = Specification(fence,x->(x[2] < 0 && x[1]<=dom_length/2))
    #sky=Specification(fence,x->(x[2]>dom_height && x[1]<=dom_length/2))
    #wind=Specification(fence,x->((x[1]<=-dom_length/2) && (x[2]>=0 && x[2]<=dom_height)))
    #mountain=Witch(hₘ,a)
    sys = ParticleSystem(Particle,domain+fence, h)
    generate_particles!(sys,grid,domain,x -> Particle(x,VEC0,FLUID))
    generate_particles!(sys,grid,fence,x -> Particle(x,VEC0,WALL))
    #generate_particles!(sys,grid,wind,x -> Particle(x,VEC0,INFLOW))
    #generate_particles!(sys,grid,mountain,x -> Particle(x,VEC0,OBSTACLE))
    #create_cell_list!(sys)
    #apply!(sys,find_pressure!)
	return sys
end

#=
### Deploy SPH equations
=#

@inbounds function balance_of_mass!(p::Particle, q::Particle, r::Float64)
	ker =q.m*rDwendland2(h,r)
	p.Drho += ker*(dot(p.x-q.x, p.u-q.u))
    #if (p.type == FLUID && q.type == FLUID) || (p.type==FLUID && q.type==INFLOW)
    #    p.Drho += 2*nu/p.rho*(p.rho - q.rho)
    #end
end

@inbounds function internal_force!(p::Particle, q::Particle, r::Float64)
	ker = q.m*rDwendland2(h,r)
    x_pq = p.x - q.x
	p.Du += -ker*(p.P/p.rho^2 + q.P/q.rho^2)*x_pq
    #p.Du += 8.0*ker*mu/(p.rho*q.rho)*dot(p.u - q.u, x_pq)/(r*r + 0.01*h*h)*x_pq
end

#=
### Set the initial density 
=#

function set_density!(p::Particle)
    p.rho=rho0*exp(-p.x[2]*g/(R_gas*T))
end

#=
### Calculate pressure
=#

function find_pressure!(p::Particle)
    #if p.x[1] >= -dom_length/2 + h
	#     p.rho += p.Drho*dt
    #end
    p.rho+=p.Drho*dt
	p.Drho = 0.0
	p.P = p.rho*R_gas*T   
end

#=
### Generate new particles
=#

function add_new_particles!(sys::ParticleSystem)
    new_particles = Particle[]
    for p in sys.particles
        if p.type == INFLOW && p.x[1] >= -dom_length/2
            p.type = FLUID
            x = p.x - bc_width*VECX
            newp = Particle(x,U_max*VECX,INFLOW)
            push!(new_particles, newp)
        end
    end
    append!(sys.particles, new_particles)
end

function set_inflow_speed!(p::Particle)
    if p.type == INFLOW
        p.u = U_max*VECX
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
	if p.type == FLUID || p.type == INFLOW
		p.x += dt*p.u
	end
end

function accelerate!(p::Particle)
	if p.type == FLUID 
		p.u += 0.5*dt*(p.Du - g*VECY) #(-damping_structure(p.x[2],zₜ,zᵦ,γᵣ)*VECY-g*VECY+p.Du)
	end
end


function  main()
    sys = make_system()
	out = new_pvd_file(folder_name)
    save_frame!(out, sys, export_vars...)
    nsteps = Int64(round(t_end/dt))
    nsamples = 0
    obstacle = filter(p -> p.type==OBSTACLE, sys.particles)
    #a modified Verlet scheme
	for k = 1 : nsteps 
        t = k*dt
        apply!(sys, accelerate!)
        apply!(sys, move!)
        #add_new_particles!(sys)
        #apply!(sys,set_inflow_speed!)
        create_cell_list!(sys)
		apply!(sys, balance_of_mass!)
        apply!(sys, find_pressure!)
        apply!(sys, internal_force!)
        apply!(sys, accelerate!)

        #save data at selected 
        if (k %  Int64(round(dt_frame/dt)) == 0)
            @show t
            println("N = ", length(sys.particles))
            save_frame!(out, sys, export_vars...)
        end
	end
	save_pvd_file(out)
end

end



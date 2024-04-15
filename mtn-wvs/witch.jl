#=

# Flow around a mountain with the Witch of Agnesi profile:

h(x)=(hₘa²)/(x²+a²)

=#

module witch

using Printf
using SmoothedParticles

const folder_name = "results/witch"

#=
Declare constants
=#

#geometry parameters
scale=1e4                     #scale to make simulation smaller for testing
const dom_height = 26e3/scale #height of the domain 
const dom_length = 400e3/scale#length of the domain
const dr = dom_length/1000	 #average particle distance (decrease to make finer simulation)
const h = 2.4*dr              
const bc_width = 6*dr       

const H = 100/scale           #parameters for the Witch of Agnesi profile; mountain height
const a = 10e3/scale          #parameters for the Witch of Agnesi profile; mountain width



#physical parameters
const U_max = 20.0       #maximum inflow velocity
const rho0 =1.177		#referential fluid density
const m0 = rho0*dr^2	#particle mass
const c = 20.0*U_max	#numerical speed of sound
#const mu = 1.0e-3		#dynamic viscosity
#const nu = 0.1*h*c      #pressure stabilization


#temporal parameters
const dt = 0.1*h/c                     #time step
const t_end = 5                       #end of simulation, 4*3600
const dt_frame = max(dt, t_end/200)    #how often data is saved

#particle types
const FLUID = 0.0
const INFLOW = 1.0
const WALL = 2.0
const OBSTACLE = 3.0

#=
Declare variables to be stored in a Particle
=#
mutable struct Particle <: AbstractParticle
    x::RealVector #position
    v::RealVector #velocity
    a::RealVector #acceleration
    rho::Float64 #density
    Drho::Float64 #rate of density
    P::Float64 #pressure
    m::Float64 #mass
    type::Float64 #particle type
    Particle(x, type=FLUID) = begin
        return new(x, VEC0, VEC0,  rho0, 0.0, 0.0, m0, type)
    end
end

#=
### Define geometry and create particles
=#

function make_system()
    grid=Grid(dr,:exp,0.0196)
    domain = Rectangle(0.0,0.0, dom_length, dom_height)
    wall=BoundaryLayer(domain,grid,bc_width)
    sys = ParticleSystem(Particle, domain + wall, h)
    ground = Specification(wall,x,x->x[2]=0)
    mountain=Witch(H,a)
    generate_particles(sys,grid,domain,x -> Particle(x=x, type=FLUID))
    generate_particles(sys,grid,ground,x -> Particle(x=x, type=WALL))
    generate_particles(sys,grid,mountain,x -> Particle(x=x, type=OBSTACLE))
    create_cell_list!(sys)
	apply!(sys, find_pressure!)
	apply!(sys, internal_force!) 
	return sys
end

#Inflow function

function set_inflow_speed!(p::Particle, t::Float64)
    if p.type == INFLOW
        p.v = v1*U_max
    end
end
@inbounds function balance_of_mass!(p::Particle, q::Particle, r::Float64)
    if p.type == FLUID && q.type == FLUID
        p.Drho += m*rDwendland2(h,r)*(dot(p.x-q.x, p.v-q.v))
    end
end

function find_pressure!(p::Particle)
    if p.x[1] >= -bc_width + h
	     p.rho += p.Drho*dt
    end
	p.Drho = 0.0
	p.P = c^2*(p.rho - rho0)
end

@inbounds function internal_force!(p::Particle, q::Particle, r::Float64)
	ker = q.m*rDwendland2(h,r)
    x_pq = p.x - q.x
	p.a += -ker*(p.P/p.rho^2 + q.P/q.rho^2)*x_pq
    p.a += 8.0*ker*mu/(p.rho*q.rho)*dot(p.v - q.v, x_pq)/(r*r + 0.01*h*h)*x_pq
end

function move!(p::Particle)
	p.a = VEC0
	if p.type == FLUID || p.type == INFLOW
		p.x += dt*p.v
	end
end


function accelerate!(p::Particle)
	if p.type == FLUID
		p.v += 0.5*dt*(p.a + gravity(p))
	end
end

function add_new_particles!(sys::ParticleSystem)
    new_particles = Particle[]
    for p in sys.particles
        if p.type == INFLOW && p.x[1] >= 0
            p.type = FLUID
            x = p.x - bc_width*VECX
            newp = Particle(x, INFLOW)
            push!(new_particles, newp)
        end
    end
    append!(sys.particles, new_particles)
end

function  main()
    sys = make_system()
	out = new_pvd_file(folder_name)
    save_frame!(out, sys, :v, :P, :rho, :type)
    nsteps = Int64(round(t_end/dt))
    nsamples = 0
    obstacle = filter(p -> p.type==OBSTACLE, sys.particles)
    #a modified Verlet scheme
	for k = 1 : nsteps 
        t = k*dt
        apply!(sys, accelerate!)
        apply!(sys, move!)
        add_new_particles!(sys)
        apply!(sys, p -> set_inflow_speed!(p,t))
        create_cell_list!(sys)
		apply!(sys, balance_of_mass!)
        apply!(sys, find_pressure!)
        apply!(sys, internal_force!)
        apply!(sys, accelerate!)

        #save data at selected 
        if (k %  Int64(round(dt_frame/dt)) == 0)
            @show t
            println("N = ", length(sys.particles))
            save_frame!(out, sys, :v, :P, :rho, :type)
        end
	end
	save_pvd_file(out)
end

end



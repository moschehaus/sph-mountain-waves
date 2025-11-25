using Match

abstract type Grid end
abstract type Grid2 <: Grid end
abstract type Grid3 <: Grid end

const golden_angle = 2.39996322972865332

"""
    Grid(dr::Float64, symm::Symbol)::Grid

Define a grid with a given characterstic length ``\\text{d}r`` and symmetry type
`symm`. Characterstic length means that one particle occupies a volume
``\\text{d}r^n``, where ``n`` is the dimension. Supported symmetry types are

2D:
1) `:square` (square grid)
2) `:hexagonal` (hexagrid, dual of isometric grid)
3) `:vogel` (grid based on Vogel's spiral, see: https://en.wikipedia.org/wiki/Fermat%27s_spiral)
4) `:exp` (exponentially distributed particles)

3D:
1) `:cubic` (cubic grid, see: https://en.wikipedia.org/wiki/Cubic_crystal_system)
2) `:facecentered` (cubic face-centered grid)
3) `:bodycentered` (cubic body-centered grid)
4) `:diamond` (diamond cubic grid, see: https://en.wikipedia.org/wiki/Diamond_cubic)
"""
function Grid(dr::Float64, symm::Symbol; K::Float64)::Grid
        return @match symm begin
                :square => Squaregrid(dr)
                :hexagonal => Hexagrid(dr)
                :vogel => VogelGrid(dr)
                :exp => ExpGrid(dr, K)
                :cubic => CubicGrid(dr)
                :facecentered => FacecenteredGrid(dr)
                :bodycentered => BodycenteredGrid(dr)
                :diamond => DiamondGrid(dr)
                _ => @error("Unsupported grid type: " * string(symm))
        end
end

function dimension(::Grid2)::Int64
        return 2
end

function dimension(::Grid3)::Int64
        return 3
end

mutable struct Squaregrid <: Grid2
        dr::Float64
end

function covering(grid::Squaregrid, s::Shape)::Vector{RealVector}
        xs = RealVector[]
        rect = boundarybox(s)
        i_min = Int64(floor(rect.x1_min / grid.dr))
        j_min = Int64(floor(rect.x2_min / grid.dr))
        i_max = Int64(ceil(rect.x1_max / grid.dr))
        j_max = Int64(ceil(rect.x2_max / grid.dr))
        for i in i_min:i_max, j in j_min:j_max
                x = RealVector(i * grid.dr, j * grid.dr, 0.0)
                if is_inside(x, s)
                        push!(xs, x)
                end
        end
        return xs
end

mutable struct Hexagrid <: Grid2
        dr::Float64
        a::Float64
        b::Float64
        Hexagrid(dr::Float64) = new(dr, (4 / 3)^(1 / 4) * dr, (3 / 4)^(1 / 4) * dr)
end

function covering(grid::Hexagrid, s::Shape)::Vector{RealVector}
        xs = RealVector[]
        rect = boundarybox(s)
        i_min = Int64(floor(rect.x1_min / grid.a)) - 1
        j_min = Int64(floor(rect.x2_min / grid.b))
        i_max = Int64(ceil(rect.x1_max / grid.a))
        j_max = Int64(ceil(rect.x2_max / grid.b))
        for i in i_min:i_max, j in j_min:j_max
                x1 = (i + (j % 2) / 2) * grid.a
                x2 = j * grid.b
                x = RealVector(x1, x2, 0.0)
                if is_inside(x, s)
                        push!(xs, x)
                end
        end
        return xs
end

mutable struct VogelGrid <: Grid2
        center::RealVector
        k::Float64
        dr::Float64
        VogelGrid(dr::Float64) = begin
                k = dr / sqrt(pi)
                return new(RealVector(0.0, 0.0, 0.0), k, dr)
        end
end

function covering(vg::VogelGrid, s::Shape)::Vector{RealVector}
        bb = boundarybox(s)
        dl = RealVector(bb.x1_min, bb.x2_min, 0.0)
        dr = RealVector(bb.x1_max, bb.x2_min, 0.0)
        ur = RealVector(bb.x1_max, bb.x2_max, 0.0)
        ul = RealVector(bb.x1_min, bb.x2_max, 0.0)
        R = 0.0
        for corner in [dl, dr, ur, ul]
                R = max(R, norm(corner - vg.center))
        end
        N = (R / vg.k)^2
        xs = RealVector[]
        for n in 1:N
                x = vg.center + vg.k * sqrt(n) * RealVector(cos(n * golden_angle), sin(n * golden_angle), 0.0)
                if is_inside(x, s)
                        push!(xs, x)
                end
        end
        return xs
end

"""
ExpGrid
"""

mutable struct ExpGrid <: Grid2
        dr::Float64
        K::Float64
end

function covering(grid::ExpGrid, s::Shape)::Vector{RealVector}
        rect = boundarybox(s)
        dr = grid.dr
        K = grid.K

        x_min = rect.x1_min
        x_max = rect.x1_max
        y_min = rect.x2_min
        r_max = rect.x2_max
        H = r_max - y_min

        # choose number of vertical layers so that average spacing ≈ dr
        Nz = max(1, Int(round(H / dr)))

        # horizontal index range (like Squaregrid)
        i_min = Int(floor(x_min / dr))
        i_max = Int(ceil(x_max / dr))

        e_min = exp(-K * y_min)
        e_max = exp(-K * r_max)

        xs = RealVector[]

        for j in 0:(Nz-1)
                # s in (0,1), midpoint of each layer
                s_frac = (j + 0.5) / Nz

                # invert truncated exponential CDF
                e_y = e_min - s_frac * (e_min - e_max)
                y = -log(e_y) / K

                for i in i_min:i_max
                        x = RealVector(i * dr, y, 0.0)
                        if is_inside(x, s)
                                push!(xs, x)
                        end
                end
        end
        return xs
end

mutable struct CubicGrid <: Grid3
        dr::Float64
end

function covering(grid::CubicGrid, s::Shape)::Vector{RealVector}
        xs = RealVector[]
        box = boundarybox(s)
        i_min = Int64(floor(box.x1_min / grid.dr))
        j_min = Int64(floor(box.x2_min / grid.dr))
        k_min = Int64(floor(box.x3_min / grid.dr))
        i_max = Int64(ceil(box.x1_max / grid.dr))
        j_max = Int64(ceil(box.x2_max / grid.dr))
        k_max = Int64(ceil(box.x3_max / grid.dr))
        for i in i_min:i_max, j in j_min:j_max, k in k_min:k_max
                x = RealVector(i * grid.dr, j * grid.dr, k * grid.dr)
                if is_inside(x, s)
                        push!(xs, x)
                end
        end
        return xs
end

mutable struct BodycenteredGrid <: Grid3
        dr::Float64
end

function covering(grid::BodycenteredGrid, s::Shape)::Vector{RealVector}
        xs = RealVector[]
        box = boundarybox(s)
        a = 2^(1 / 3) * grid.dr
        i_min = Int64(floor(box.x1_min / a))
        j_min = Int64(floor(box.x2_min / a))
        k_min = Int64(floor(box.x3_min / a))
        i_max = Int64(ceil(box.x1_max / a))
        j_max = Int64(ceil(box.x2_max / a))
        k_max = Int64(ceil(box.x3_max / a))
        for i in i_min:i_max, j in j_min:j_max, k in k_min:k_max
                x = RealVector(i * a, j * a, k * a)
                if is_inside(x, s)
                        push!(xs, x)
                end
        end
        for i in i_min:i_max, j in j_min:j_max, k in k_min:k_max
                x = RealVector((i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a)
                if is_inside(x, s)
                        push!(xs, x)
                end
        end
        return xs
end

mutable struct FacecenteredGrid <: Grid3
        dr::Float64
end

function covering(grid::FacecenteredGrid, s::Shape)::Vector{RealVector}
        xs = RealVector[]
        box = boundarybox(s)
        a = 4^(1 / 3) * grid.dr
        i_min = Int64(floor(box.x1_min / a))
        j_min = Int64(floor(box.x2_min / a))
        k_min = Int64(floor(box.x3_min / a))
        i_max = Int64(ceil(box.x1_max / a))
        j_max = Int64(ceil(box.x2_max / a))
        k_max = Int64(ceil(box.x3_max / a))
        for i in i_min:i_max, j in j_min:j_max, k in k_min:k_max
                x = RealVector(i * a, j * a, k * a)
                if is_inside(x, s)
                        push!(xs, x)
                end
        end
        for i in i_min:i_max, j in j_min:j_max, k in k_min:k_max
                x = RealVector((i + 0.5) * a, (j + 0.5) * a, k * a)
                if is_inside(x, s)
                        push!(xs, x)
                end
                x = RealVector((i + 0.5) * a, j * a, (k + 0.5) * a)
                if is_inside(x, s)
                        push!(xs, x)
                end
                x = RealVector(i * a, (j + 0.5) * a, (k + 0.5) * a)
                if is_inside(x, s)
                        push!(xs, x)
                end
        end
        return xs
end

mutable struct DiamondGrid <: Grid3
        dr::Float64
end

function covering(grid::DiamondGrid, s::Shape)::Vector{RealVector}
        xs = RealVector[]
        box = boundarybox(s)
        a = 0.5 * grid.dr
        i_min = Int64(floor(box.x1_min / a))
        j_min = Int64(floor(box.x2_min / a))
        k_min = Int64(floor(box.x3_min / a))
        i_max = Int64(ceil(box.x1_max / a))
        j_max = Int64(ceil(box.x2_max / a))
        k_max = Int64(ceil(box.x3_max / a))
        for i in i_min:i_max, j in j_min:j_max, k in k_min:k_max
                if isodd(i) == isodd(j) == isodd(k)
                        sum = (i + j + k) % 4
                        sum = (sum + 4) % 4   #we want positive number after moduling (like every normal person)
                        if sum == 0 || sum == 1
                                x = RealVector(i * a, j * a, k * a)
                                if is_inside(x, s)
                                        push!(xs, x)
                                end
                        end
                end
        end
        return xs
end



"""
    generate_particles!(sys::ParticleSystem,
                        grid::Grid,
                        geometry::Shape,
                        constructor::Function)

Create particles using `constructor(x::RealVector)::AbstractParticle` at every `grid` point
inside a given shape.

"""
function generate_particles!(sys::ParticleSystem, grid::Grid, geometry::Shape, constructor::Function)
        xs = covering(grid, geometry)
        for x in xs
                push!(sys.particles, constructor(x))
        end
end


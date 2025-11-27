const dt_pack = 1.0 * dt
const c_pack = 2.0 * c  # or maybe 0.5c or 2c; tune later
const ζ_pack = 1.0 * c / dt_pack   # for example; tune this

@inbounds function reset_rho_pack!(p::AbstractParticle)
        if p.type == FLUID
                p.ρ = 0.0
        end
end

@inbounds function accumulate_rho_pack!(p::AbstractParticle, q::AbstractParticle, r::Float64)
        if p.type == FLUID
                p.ρ += q.m * wendland2(p.h, r)
        end
end

# target hydrostatic density profile 
@inline function rho_target(z::Float64)
        return ρ0 * exp(-z * g / (R_mass * T_bg))
end

# ----- packing forcing -----
@inbounds function balance_of_momentum_pack!(p::AbstractParticle, q::AbstractParticle, r::Float64)
        if p.type == FLUID && q.type == FLUID
                x_pq = p.x - q.x

                ρi = max(p.ρ, rho_floor)
                ρj = max(q.ρ, rho_floor)

                ρti = rho_target(p.x[2])
                ρtj = rho_target(q.x[2])

                Pi = c_pack^2 * (ρi - ρti)
                Pj = c_pack^2 * (ρj - ρtj)

                ker = rDwendland2(0.5 * (p.h + q.h), r)

                f = -q.m * (Pi / ρi^2 + Pj / ρj^2) * ker * x_pq

                # *** move only in vertical (y) direction  ***

                p.Dv += f[2] * VECY
        end
end


# strong damping
@inbounds function packing_accelerate!(p::AbstractParticle)
        if p.type == FLUID
                v_old = p.v
                F = p.Dv
                p.v = (v_old + dt_pack * F) / (1.0 + ζ_pack * dt_pack)
        end
        p.Dv = VEC0
end

@inbounds function packing_move!(p::AbstractParticle)
        if p.type == FLUID
                p.x += dt_pack * p.v
        end
end

# main loop
function packing!(sys::ParticleSystem;
        abs_tol=1e-3,
        rel_tol=1e-2,
        maxSteps=500)

        # reset velocities
        for p in sys.particles
                p.v = VEC0
                p.Dv = VEC0
        end

        # initial density
        apply!(sys, reset_rho_pack!)
        apply!(sys, accumulate_rho_pack!)

        # measure initial residual
        ρ_err0 = 0.0
        for p in sys.particles
                if p.type == FLUID
                        ρt = rho_target(p.x[2])
                        ρ_err0 += (p.ρ - ρt)^2
                end
        end
        ρ_err0 = sqrt(ρ_err0)

        println("---- PACKING INIT ----")
        println("Initial density error = $ρ_err0")

        k = 0
        while k < maxSteps
                # pseudo-step
                apply!(sys, packing_accelerate!)
                apply!(sys, packing_move!)
                create_cell_list!(sys)

                # recompute density
                apply!(sys, reset_rho_pack!)
                apply!(sys, accumulate_rho_pack!)

                # packing forces
                apply!(sys, balance_of_momentum_pack!)
                apply!(sys, packing_accelerate!)

                # show diagnostics every N steps
                if k % 10 == 0
                        ρ_err = 0.0
                        v_norm2 = 0.0
                        for p in sys.particles
                                if p.type == FLUID
                                        ρt = rho_target(p.x[2])
                                        ρ_err += (p.ρ - ρt)^2
                                        v_norm2 += dot(p.v, p.v)
                                end
                        end
                        ρ_err = sqrt(ρ_err)
                        v_norm = sqrt(v_norm2)

                        crit = abs_tol + rel_tol * ρ_err0
                        println("packing step $k: ρ_err = $ρ_err, |v| = $v_norm, crit = $crit")

                        if (ρ_err < crit) && (v_norm < crit)
                                break
                        end
                end

                k += 1
        end

        # set velocities to zero after packing
        for p in sys.particles
                p.v = VEC0
                p.Dv = VEC0
        end

        println("---- PACKING DONE AFTER $k STEPS ----")
        return sys
end


valid(s,m) = abs(m) <= s || isapprox(s, m, atol=1e-5)
mvals(N) = range(-N/2, N/2)
svals(N) = collect(0:N÷2)

function P_JM_z_minus2(J, M, N)
    if J == 0 return 0.0 end
    return (N + 2 * J + 2) * (J - M) * (J + M) / (4 * J * (2 * J + 1))
end
A_JM_minus2(J, M) = (J + M) * (J - M + 1)
P_JM_z_plus2(J, M, N) = (N - 2 * J) * (J + 1 - M) * (J + 1 + M) / (4 * (J + 1) * (2 * J + 1))

function P_JM_minus_02(J, M, N)
    if J == 0
        return 0.0
    end
    return ((2 + N) / (4 * J * (J + 1))) * A_JM_minus2(J, M)
end

function P_JM_minus_minus2(J, M, N) 
    if J == 0
        return 0.0
    end
    return ((N + 2 * J + 2) * (J + M) * (J + M - 1) / (4 * J * (2 * J + 1)))
end

P_JM_minus_plus2(J, M, N) = ((N - 2 * J) * (J - M + 1) * (J - M + 2) / (4 * (J + 1) * (2 * J + 1)))

function run_trajectory(N::Int64, Γ::Float64, ξ::Float64, γ::Float64, t_max::Float64, observable; M0::Int64 = N ÷ 2, S0::Int64 = N ÷ 2, sizehint::Int64 = 0)
    
    S::Int64 = S0
    M::Int64 = M0
    t = 0.0
    
    times::Vector{Float64} = []; sizehint!(times, sizehint)
    out::Vector{typeof(observable(S, M, N, Γ, ξ))} = []; sizehint!(out, sizehint)

    push!(times, 0.0)
    push!(out, observable(S, M, N, Γ, ξ))

    
    while t < t_max
        # 1. Calculate rates
        r_dec = Γ * A_JM_minus2(S, M) + γ * P_JM_minus_02(S, M, N)
        r_up  = ξ * P_JM_z_plus2(S, M, N)
        r_down = ξ * P_JM_z_minus2(S, M, N)
        r_dec_down = γ * P_JM_minus_minus2(S, M, N)
        r_dec_up = γ * P_JM_minus_plus2(S, M, N)
        
        r_total = r_dec + r_up + r_down + r_dec_up + r_dec_down
        
        if r_total <= 1e-10
            push!(times, t_max)
            push!(out, observable(S, M, N, Γ, ξ))
            break
        end
        

        tau = -log(rand()) / r_total        
        r_val = rand() * r_total
        
        if r_val < r_dec
            # Decay: m -> m - 1
            M -= 1
        elseif r_val < r_dec + r_up
            # Dephasing Up: s -> s + 1
            S += 1
        elseif r_val < r_dec + r_up + r_down
            # Dephasing Down: s -> s - 1
            S -= 1
        elseif r_val < r_dec + r_up + r_down + r_dec_up
            #
            S += 1
            M -= 1
        else
            #
            S -= 1
            M -= 1
        end
        
        t += tau        
        push!(times, t)
        push!(out, observable(S, M, N, Γ, ξ))
    end

    return times, out
end

function simulate_ensemble_single(N, Γ, ξ, γ, t_max::Float64, observable, N_traj; M0 = N ÷ 2, S0 = N ÷ 2)
    
    # Create a common time grid for averaging
    time_grid = range(0, t_max, length=1000)
    avg_obs = Vector{Any}(nothing, length(time_grid))

    for i=1:N_traj
        ts, Is= run_trajectory(N, Γ, ξ, γ, t_max, observable; M0 = M0, S0 = S0)
        
        idx = 1
        for (k, t_point) in enumerate(time_grid)
            while idx < length(ts) && ts[idx+1] < t_point
                idx += 1
            end
            if avg_obs[k] == nothing
                avg_obs[k] = Is[idx]
            else
                avg_obs[k] += Is[idx]
            end
        end
    end
    
    return time_grid, avg_obs ./ N_traj
end

function der2(A, dx)
    return (A[3:end] .- 2 .* A[2:end-1] .+ A[1:end-2]) ./ dx^2
end

function der(A, dx)
    return (A[3:end] .- A[1:end-2]) ./ (2dx)
end

function der3(A, dx)
    return (A[5:end] .- 2 .* A[4:end-1] .+ 2 .* A[2:end-3] .- A[1:end-4]) ./ (2dx^3)
end

function meanfield_reduced!(du, u, p, t)
    I, m = u
    Γ, ξ, N, γ = p

    du[1] = (2 * N * Γ * m - ξ - γ) * I
    du[2] = -N * Γ * I - γ * (m + 0.5)
end

function mc_vs_mf_data(N, params, N_traj; time_factor = 3.0)
    tmax = time_factor * log(N) / N
    times_mc, intens_mc = simulate_ensemble_single(N, params["global_decay"], params["local_dephasing"], params["local_decay"], tmax, (S, M, N, Γ, ξ) -> (S+M) * (S-M+1), N_traj)
    # mean field equation of motion
    p = [params["global_decay"], params["local_dephasing"], N, params["local_decay"]]
    u0 = [1/N, 1/2]


    prob = ODEProblem(meanfield_reduced!, u0, (0.0, tmax), p)
    mfsol = solve(prob, AutoTsit5(Rosenbrock23()), reltol=1e-8, abstol=1e-8, saveat = range(0.0, tmax, 20000))
    intens_mf = map(x -> x[1], mfsol.u)
    return maximum(intens_mc), maximum(intens_mf), times_mc[argmax(intens_mc)], mfsol.t[argmax(intens_mf)]
end

function mc_data(N, params, N_traj; time_factor = 3.0)

    tmax = time_factor * log(N) / N
    times_mc, intens_mc = simulate_ensemble_single(N, params["global_decay"], params["local_dephasing"], params["local_decay"], tmax, (S, M, N, Γ, ξ) -> (S+M) * (S-M+1), N_traj)
    return maximum(intens_mc), times_mc[argmax(intens_mc)]
end

function mf_data(N, params; time_factor = 3.0)
    tmax = time_factor * log(N) / N
    # mean field equation of motion
    p = [params["global_decay"], params["local_dephasing"], N, params["local_decay"]]
    u0 = [1/N, 1/2]
    prob = ODEProblem(meanfield_reduced!, u0, (0.0, tmax), p)
    mfsol = solve(prob, AutoTsit5(Rosenbrock23()), reltol=1e-8, abstol=1e-8)
    intens_mf = map(x -> x[1], mfsol.u)
    return maximum(intens_mf),  mfsol.t[argmax(intens_mf)]
end

expval_S(state, index_mapping) = sum(state[value] * key[1] for (key,value) in index_mapping)
expval_M(state, index_mapping) = sum(state[value] * key[2] for (key,value) in index_mapping)
expval_I(state, index_mapping) = sum(state[value] * ((S+M) * (S-M+1)) for ((S,M),value) in index_mapping)

function simulate_ensemble(N, Γ, ξ, γ, t_max::Float64, observable, N_traj; M0 = N ÷ 2, S0 = N ÷ 2)

    time_grid = range(0, t_max, length=10000)

    # First trajectory to infer observable shape/type
    ts, Is = run_trajectory(N, Γ, ξ, γ, t_max, observable; M0=M0, S0=S0)
    sample = Is[1]

    avg_obs = [zero(sample) for _ in eachindex(time_grid)]

    # helper to add one trajectory onto the common grid
    function accumulate!(avg_obs, ts, Is, time_grid)
        idx = 1
        for (k, t_point) in enumerate(time_grid)
            while idx < length(ts) && ts[idx+1] < t_point
                idx += 1
            end
            avg_obs[k] .+= Is[idx]
        end
    end

    accumulate!(avg_obs, ts, Is, time_grid)

    for i in 2:N_traj
        ts, Is = run_trajectory(N, Γ, ξ, γ, t_max, observable; M0=M0, S0=S0)
        accumulate!(avg_obs, ts, Is, time_grid)
    end

    return time_grid, [x / N_traj for x in avg_obs]
end

element(arr, n) = map(x->x[n], arr)

function sci_label(N)
    exp = floor(Int, log10(N))
    mant = N / 10.0^exp

    # if mantissa is (numerically) an integer → print as Int
    if isapprox(mant, round(mant); atol=1e-4)
        mant_str = string(Int(round(mant)))
    else
        mant_str = string(mant)
    end

    return "\$N=$(mant_str)\\times 10^{$exp}\$"
end

function namedtuple_to_dict(nt::NamedTuple)
    Dict(string(k) => v for (k, v) in pairs(nt))
end

function meanfield_log_transformed!(du, u, p, t)
    L, m = u  # u[1] is log(I)
    Γ, ξ, N, γ = p

    du[1] = (2 * N * Γ * m - ξ - γ)
    I_current = exp(L)
    du[2] = -N * Γ * I_current - γ * (m + 0.5)
end
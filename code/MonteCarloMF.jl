using Random
using OrdinaryDiffEq

valid(s,m) = abs(m) <= s || isapprox(s, m, atol=1e-5)
mvals(N) = range(-N/2, N/2)
svals(N) = collect(0:N÷2)

@inline function P_JM_z_minus2(J::Float64, M::Float64, N::Float64)
    J == 0.0 && return 0.0
    return (N + 2.0 * J + 2.0) * (J - M) * (J + M) /
           (4.0 * J * (2.0 * J + 1.0))
end

@inline A_JM_minus2(J::Float64, M::Float64) =
    (J + M) * (J - M + 1.0)

@inline function P_JM_z_plus2(J::Float64, M::Float64, N::Float64)
    return (N - 2.0 * J) * (J + 1.0 - M) * (J + 1.0 + M) /
           (4.0 * (J + 1.0) * (2.0 * J + 1.0))
end

@inline function P_JM_minus_02(J::Float64, M::Float64, N::Float64)
    J == 0.0 && return 0.0
    return (N + 2.0) * (J + M) * (J - M + 1.0) /
           (4.0 * J * (J + 1.0))
end

@inline function P_JM_minus_minus2(J::Float64, M::Float64, N::Float64)
    J == 0.0 && return 0.0
    return (N + 2.0 * J + 2.0) * (J + M) * (J + M - 1.0) /
           (4.0 * J * (2.0 * J + 1.0))
end

@inline function P_JM_minus_plus2(J::Float64, M::Float64, N::Float64)
    return (N - 2.0 * J) * (J - M + 1.0) * (J - M + 2.0) /
           (4.0 * (J + 1.0) * (2.0 * J + 1.0))
end

function run_trajectory(N::Int64, Γ::Float64, ξ::Float64, γ::Float64, t_max::Float64, observable; M0::Int64 = N ÷ 2, S0::Int64 = N ÷ 2, sizehint::Int64 = 0)
    
    S::Float64 = S0
    M::Float64 = M0
    Nfloat::Float64 = N
    t = 0.0
    
    times::Vector{Float64} = []; sizehint!(times, sizehint)
    out::Vector{typeof(observable(S, M, Nfloat, Γ, ξ))} = []; sizehint!(out, sizehint)

    push!(times, 0.0)
    push!(out, observable(S, M, Nfloat, Γ, ξ))

    
    while t < t_max
        # 1. Calculate rates
        r_dec = Γ * A_JM_minus2(S, M) + γ * P_JM_minus_02(S, M, Nfloat)
        r_up  = ξ * P_JM_z_plus2(S, M, Nfloat)
        r_down = ξ * P_JM_z_minus2(S, M, Nfloat)
        r_dec_down = γ * P_JM_minus_minus2(S, M, Nfloat)
        r_dec_up = γ * P_JM_minus_plus2(S, M, Nfloat)
        
        r_total = r_dec + r_up + r_down + r_dec_up + r_dec_down
        
        if r_total <= 1e-10
            push!(times, t_max)
            push!(out, observable(S, M, Nfloat, Γ, ξ))
            break
        end
        

        tau = -log(rand()) / r_total        
        r_val = rand() * r_total
        
        if r_val < r_dec
            # Decay: m -> m - 1
            M -= 1.0
        elseif r_val < r_dec + r_up
            # Dephasing Up: s -> s + 1
            S += 1.0
        elseif r_val < r_dec + r_up + r_down
            # Dephasing Down: s -> s - 1
            S -= 1.0
        elseif r_val < r_dec + r_up + r_down + r_dec_up
            #
            S += 1.0
            M -= 1.0
        else
            #
            S -= 1.0
            M -= 1.0
        end
        
        t += tau        
        push!(times, t)
        push!(out, observable(S, M, Nfloat, Γ, ξ))
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

function meanfield_log_transformed!(du, u, p, t)
    L, m = u  # u[1] is log(I)
    Γ, ξ, N, γ = p

    du[1] = (2 * N * Γ * m - ξ - γ)
    I_current = exp(L)
    du[2] = -N * Γ * I_current - γ * (m + 0.5)
end


function run_trajectory_sampled(
    NI::Int,
    Γ::Float64,
    ξ::Float64,
    γ::Float64,
    saveat::AbstractVector{<:Real},
    observable;
    M0::Int = NI ÷ 2,
    S0::Int = NI ÷ 2,
    rng = Random.default_rng(),
)
    S::Float64 = S0
    M::Float64 = M0
    N::Float64 = NI
    t = 0.0

    out = Vector{Float64}(undef, length(saveat))
    isave = 1

    while isave <= length(saveat) && saveat[isave] <= 0.0
        @inbounds out[isave] = observable(S, M, N, Γ, ξ)
        isave += 1
    end

    while isave <= length(saveat)
        r_dec = Γ * A_JM_minus2(S, M) + γ * P_JM_minus_02(S, M, N)
        r_up = ξ * P_JM_z_plus2(S, M, N)
        r_down = ξ * P_JM_z_minus2(S, M, N)
        r_dec_down = γ * P_JM_minus_minus2(S, M, N)
        r_dec_up = γ * P_JM_minus_plus2(S, M, N)

        r_total = r_dec + r_up + r_down + r_dec_up + r_dec_down

        if r_total <= 1e-10
            value = observable(S, M, N, Γ, ξ)
            @inbounds for k in isave:length(saveat)
                out[k] = value
            end
            break
        end

        t_next = t - log(rand(rng)) / r_total
        value = observable(S, M, N, Γ, ξ)

        # The state remains unchanged on [t, t_next).
        while isave <= length(saveat) && saveat[isave] < t_next
            @inbounds out[isave] = value
            isave += 1
        end

        isave > length(saveat) && break

        r_val = rand(rng) * r_total

        if r_val < r_dec
            M -= 1
        elseif r_val < r_dec + r_up
            S += 1
        elseif r_val < r_dec + r_up + r_down
            S -= 1
        elseif r_val < r_dec + r_up + r_down + r_dec_up
            S += 1
            M -= 1
        else
            S -= 1
            M -= 1
        end

        t = t_next
    end

    return out
end

function mc_data_sampled(N, params, N_traj; time_factor = 3.0, nsave = 2000)
    Γ = params["global_decay"]
    ξ = params["local_dephasing"]
    γ = params["local_decay"]

    tmax = time_factor * log(N) / N
    saveat = collect(range(0.0, tmax, length = nsave))
    rng = Xoshiro(1234)

    intens_mc = zeros(Float64, nsave)

    @inbounds for _ in 1:N_traj
        intens_mc .+= run_trajectory_sampled(
            N,
            Γ,
            ξ,
            γ,
            saveat,
            (S, M, N, Γ, ξ) -> (S + M) * (S - M + 1),
            rng = rng
        )
    end

    intens_mc ./= N_traj

    i_peak = argmax(intens_mc)
    return intens_mc[i_peak], saveat[i_peak]
end

function mc_vs_mf_data_sampled(N, params, N_traj; time_factor = 3.0, nsave = 2000)
    tmax = time_factor * log(N) / N

    times_mc, intens_mc = mc_data_sampled(
        N,
        params,
        N_traj;
        time_factor = time_factor,
        nsave = nsave,
    )

    p = [
        params["global_decay"],
        params["local_dephasing"],
        N,
        params["local_decay"],
    ]

    u0 = [1 / N, 1 / 2]

    prob = ODEProblem(meanfield_reduced!, u0, (0.0, tmax), p)

    mfsol = solve(
        prob,
        AutoTsit5(Rosenbrock23());
        reltol = 1e-8,
        abstol = 1e-8,
        saveat = times_mc,
    )

    intens_mf = getindex.(mfsol.u, 1)

    i_mc = argmax(intens_mc)
    i_mf = argmax(intens_mf)

    return intens_mc[i_mc], intens_mf[i_mf], times_mc[i_mc], times_mc[i_mf]
end

function run_trajectory_sampled_adaptive(
    NI::Int,
    Γ::Float64,
    ξ::Float64,
    γ::Float64,
    saveat::AbstractVector{<:Real},
    observable;
    M0::Int = NI ÷ 2,
    S0::Int = NI ÷ 2,
    tail_fraction::Float64 = 1e-3,
    n_tail_points::Int = 3,
    rng = Random.default_rng(),
)
    @assert 0.0 < tail_fraction < 1.0
    @assert n_tail_points ≥ 1

    S::Float64 = S0
    M::Float64 = M0
    N::Float64 = NI
    t = 0.0

    out = Vector{Float64}(undef, length(saveat))
    isave = 1

    Imax = Ref(-Inf)
    below_count = Ref(0)
    stopped = Ref(false)

    function record_value!(i::Int, value::Float64)
        @inbounds out[i] = value

        if value > Imax[]
            Imax[] = value
            below_count[] = 0
        elseif value ≤ tail_fraction * Imax[]
            below_count[] += 1
        else
            below_count[] = 0
        end

        if below_count[] ≥ n_tail_points
            if i < length(saveat)
                @inbounds out[(i + 1):end] .= 0.0
            end
            stopped[] = true
        end

        return nothing
    end

    # Record any requested initial points.
    while isave <= length(saveat) && saveat[isave] <= 0.0
        value = Float64(observable(S, M, N, Γ, ξ))
        record_value!(isave, value)
        isave += 1

        stopped[] && return out
    end

    while isave <= length(saveat)
        r_dec = Γ * A_JM_minus2(S, M) + γ * P_JM_minus_02(S, M, N)
        r_up = ξ * P_JM_z_plus2(S, M, N)
        r_down = ξ * P_JM_z_minus2(S, M, N)
        r_dec_down = γ * P_JM_minus_minus2(S, M, N)
        r_dec_up = γ * P_JM_minus_plus2(S, M, N)

        r_total = r_dec + r_up + r_down + r_dec_up + r_dec_down

        # Genuine absorbing or numerically static state: retain its constant value.
        if r_total <= 1e-10
            value = Float64(observable(S, M, N, Γ, ξ))
            @inbounds out[isave:end] .= value
            break
        end

        t_next = t - log(rand(rng)) / r_total
        value = Float64(observable(S, M, N, Γ, ξ))

        # The state remains unchanged on [t, t_next).
        while isave <= length(saveat) && saveat[isave] < t_next
            record_value!(isave, value)
            isave += 1

            stopped[] && return out
        end

        isave > length(saveat) && break

        r_val = rand(rng) * r_total

        if r_val < r_dec
            M -= 1.0
        elseif r_val < r_dec + r_up
            S += 1.0
        elseif r_val < r_dec + r_up + r_down
            S -= 1.0
        elseif r_val < r_dec + r_up + r_down + r_dec_up
            S += 1.0
            M -= 1.0
        else
            S -= 1.0
            M -= 1.0
        end

        t = t_next
    end

    return out
end

function run_trajectory_sampled_adaptive_generic(
    NI::Int,
    Γ::Float64,
    ξ::Float64,
    γ::Float64,
    saveat::AbstractVector{<:Real},
    observable;
    M0::Int = NI ÷ 2,
    S0::Int = NI ÷ 2,
    tail_fraction::Float64 = 1e-3,
    n_tail_points::Int = 3,
    rng = Random.default_rng(),
    stop_observable = (S, M, N, Γ, ξ) -> (S + M) * (S - M + 1),
)
    @assert 0.0 < tail_fraction < 1.0
    @assert n_tail_points ≥ 1

    S::Float64 = S0
    M::Float64 = M0
    N::Float64 = NI
    t = 0.0

    sample = observable(S, M, N, Γ, ξ)
    out = Vector{typeof(sample)}(undef, length(saveat))
    isave = 1

    Imax = Ref(-Inf)
    below_count = Ref(0)
    stopped = Ref(false)

    function record_value!(i::Int, value)
        @inbounds out[i] = value

        stop_value = Float64(stop_observable(S, M, N, Γ, ξ))
        if stop_value > Imax[]
            Imax[] = stop_value
            below_count[] = 0
        elseif stop_value ≤ tail_fraction * Imax[]
            below_count[] += 1
        else
            below_count[] = 0
        end

        if below_count[] ≥ n_tail_points
            if i < length(saveat)
                @inbounds out[(i + 1):end] .= Ref(value)
            end
            stopped[] = true
        end

        return nothing
    end

    while isave <= length(saveat) && saveat[isave] <= 0.0
        record_value!(isave, observable(S, M, N, Γ, ξ))
        isave += 1

        stopped[] && return out
    end

    while isave <= length(saveat)
        r_dec = Γ * A_JM_minus2(S, M) + γ * P_JM_minus_02(S, M, N)
        r_up = ξ * P_JM_z_plus2(S, M, N)
        r_down = ξ * P_JM_z_minus2(S, M, N)
        r_dec_down = γ * P_JM_minus_minus2(S, M, N)
        r_dec_up = γ * P_JM_minus_plus2(S, M, N)

        r_total = r_dec + r_up + r_down + r_dec_up + r_dec_down

        if r_total <= 1e-10
            value = observable(S, M, N, Γ, ξ)
            @inbounds out[isave:end] .= Ref(value)
            break
        end

        t_next = t - log(rand(rng)) / r_total
        value = observable(S, M, N, Γ, ξ)

        while isave <= length(saveat) && saveat[isave] < t_next
            record_value!(isave, value)
            isave += 1

            stopped[] && return out
        end

        isave > length(saveat) && break

        r_val = rand(rng) * r_total

        if r_val < r_dec
            M -= 1
        elseif r_val < r_dec + r_up
            S += 1
        elseif r_val < r_dec + r_up + r_down
            S -= 1
        elseif r_val < r_dec + r_up + r_down + r_dec_up
            S += 1
            M -= 1
        else
            S -= 1
            M -= 1
        end

        t = t_next
    end

    return out
end

function run_trajectory_sampled_adaptive_generic!(
    obs,
    NI::Int,
    Γ::Float64,
    ξ::Float64,
    γ::Float64,
    saveat::AbstractVector{<:Real},
    observable;
    M0::Int = NI ÷ 2,
    S0::Int = NI ÷ 2,
    tail_fraction::Float64 = 1e-3,
    n_tail_points::Int = 3,
    rng = Random.default_rng(),
    stop_observable = (S, M, N, Γ, ξ) -> (S + M) * (S - M + 1),
)
    @assert length(obs) == length(saveat)
    @assert 0.0 < tail_fraction < 1.0
    @assert n_tail_points ≥ 1

    S::Float64 = S0
    M::Float64 = M0
    N::Float64 = NI
    t = 0.0
    isave = 1

    Imax = -Inf
    below_count = 0

    function record_value!(i, value)
        @inbounds obs[i] += value

        stop_value = Float64(stop_observable(S, M, N, Γ, ξ))

        if stop_value > Imax
            Imax = stop_value
            below_count = 0
        elseif stop_value ≤ tail_fraction * Imax
            below_count += 1
        else
            below_count = 0
        end

        return below_count ≥ n_tail_points
    end

    while isave <= length(saveat) && saveat[isave] <= 0.0
        value = observable(S, M, N, Γ, ξ)

        if record_value!(isave, value)
            return nothing
        end

        isave += 1
    end

    while isave <= length(saveat)
        r_dec = Γ * A_JM_minus2(S, M) + γ * P_JM_minus_02(S, M, N)
        r_up = ξ * P_JM_z_plus2(S, M, N)
        r_down = ξ * P_JM_z_minus2(S, M, N)
        r_dec_down = γ * P_JM_minus_minus2(S, M, N)
        r_dec_up = γ * P_JM_minus_plus2(S, M, N)

        r_total = r_dec + r_up + r_down + r_dec_up + r_dec_down

        if r_total <= 1e-10
            value = observable(S, M, N, Γ, ξ)

            @inbounds for k in isave:length(saveat)
                obs[k] += value
            end

            return nothing
        end

        t_next = t - log(rand(rng)) / r_total
        value = observable(S, M, N, Γ, ξ)

        while isave <= length(saveat) && saveat[isave] < t_next
            if record_value!(isave, value)
                return nothing
            end

            isave += 1
        end

        isave > length(saveat) && return nothing

        r_val = rand(rng) * r_total

        if r_val < r_dec
            M -= 1.0
        elseif r_val < r_dec + r_up
            S += 1.0
        elseif r_val < r_dec + r_up + r_down
            S -= 1.0
        elseif r_val < r_dec + r_up + r_down + r_dec_up
            S += 1.0
            M -= 1.0
        else
            S -= 1.0
            M -= 1.0
        end

        t = t_next
    end

    return nothing
end

function simulate_ensemble_adaptive(
    N,
    Γ,
    ξ,
    γ,
    t_max::Float64,
    observable,
    N_traj;
    M0 = N ÷ 2,
    S0 = N ÷ 2,
    nsave = 10000,
    tail_fraction = 1e-3,
    n_tail_points = 3,
    rng = Xoshiro(1234),
    stop_observable = (S, M, N, Γ, ξ) -> (S + M) * (S - M + 1),
)

    time_grid = collect(range(0.0, t_max, length = nsave))
    sample = observable(Float64(S0), Float64(M0), Float64(N), Float64(Γ), Float64(ξ))
    avg_obs = [zero(sample) for _ in eachindex(time_grid)]

    for _ in 1:N_traj
        run_trajectory_sampled_adaptive_generic!(
            avg_obs,
            N,
            Float64(Γ),
            Float64(ξ),
            Float64(γ),
            time_grid,
            observable;
            M0 = M0,
            S0 = S0,
            tail_fraction = tail_fraction,
            n_tail_points = n_tail_points,
            rng = rng,
            stop_observable = stop_observable,
        )
    end

    avg_obs ./= N_traj

    return time_grid, avg_obs
end


function mc_data_adaptive(
    N_atoms,
    params,
    N_traj;
    time_factor = 3.0,
    nsave = 2000,
    tail_fraction = 1e-3,
    n_tail_points = 3,
)
    Γ = params["global_decay"]
    ξ = params["local_dephasing"]
    γ = params["local_decay"]

    tmax = time_factor * log(N_atoms) / N_atoms
    saveat = collect(range(0.0, tmax, length = nsave))
    rng = Xoshiro(1234)

    intens_mc = zeros(Float64, nsave)

    @inbounds for _ in 1:N_traj
        intens_mc .+= run_trajectory_sampled_adaptive(
            N_atoms,
            Γ,
            ξ,
            γ,
            saveat,
            (S, M, Nobs, Γ, ξ) -> (S + M) * (S - M + 1);
            tail_fraction = tail_fraction,
            n_tail_points = n_tail_points,
            rng = rng,
        )
    end

    intens_mc ./= N_traj

    i_peak = argmax(intens_mc)
    return intens_mc[i_peak], saveat[i_peak]
end


function mc_vs_mf_adaptive(
    N_atoms,
    params,
    N_traj;
    time_factor = 3.0,
    nsave = 2000,
    tail_fraction = 1e-3,
    n_tail_points = 3,
)
    I_mc, t_mc = mc_data_adaptive(
        N_atoms,
        params,
        N_traj;
        time_factor = time_factor,
        nsave = nsave,
        tail_fraction = tail_fraction,
        n_tail_points = n_tail_points,
    )

    Γ = params["global_decay"]
    ξ = params["local_dephasing"]
    γ = params["local_decay"]

    tmax = time_factor * log(N_atoms) / N_atoms

    p = [Γ, ξ, N_atoms, γ]
    u0 = [1 / N_atoms, 1 / 2]

    prob = ODEProblem(meanfield_reduced!, u0, (0.0, tmax), p)
    mfsol = solve(
        prob,
        AutoTsit5(Rosenbrock23());
        reltol = 1e-8,
        abstol = 1e-8,
        saveat = range(0.0, tmax, length = nsave),
    )

    intens_mf = first.(mfsol.u)
    i_mf = argmax(intens_mf)

    return (
        I_mc = I_mc,
        t_mc = t_mc,
        I_mf = intens_mf[i_mf],
        t_mf = mfsol.t[i_mf],
    )
end

using Distributions

function run_trajectory_sampled_adaptive_tau(
    NI::Int,
    Γ::Float64,
    ξ::Float64,
    γ::Float64,
    saveat::AbstractVector{<:Real},
    observable;
    M0::Int = NI ÷ 2,
    S0::Int = NI ÷ 2,
    ε::Float64 = 0.03,
    min_expected_events::Float64 = 20.0,
    max_rejections::Int = 20,
    rng = Random.default_rng(),
)
    S::Float64 = S0
    M::Float64 = M0
    N::Float64 = NI
    halfN = N / 2.0
    t = 0.0

    out = Vector{Float64}(undef, length(saveat))
    isave = 1

    while isave <= length(saveat) && saveat[isave] <= 0.0
        @inbounds out[isave] = observable(S, M, N, Γ, ξ)
        isave += 1
    end

    while isave <= length(saveat)
        r_dec = Γ * A_JM_minus2(S, M) + γ * P_JM_minus_02(S, M, N)
        r_up = ξ * P_JM_z_plus2(S, M, N)
        r_down = ξ * P_JM_z_minus2(S, M, N)
        r_dec_down = γ * P_JM_minus_minus2(S, M, N)
        r_dec_up = γ * P_JM_minus_plus2(S, M, N)

        r_total = r_dec + r_up + r_down + r_dec_up + r_dec_down

        if r_total <= 1e-10
            value = observable(S, M, N, Γ, ξ)
            @inbounds for k in isave:length(saveat)
                out[k] = value
            end
            break
        end

        # Use constrained coordinates:
        #
        # x1 = S + M >= 0
        # x2 = S - M >= 0
        # x3 = N/2 - S >= 0
        #
        # Their drift and infinitesimal variance determine a safe tau step.

        x1 = S + M
        x2 = S - M
        x3 = halfN - S

        μ1 = -r_dec + r_up - r_down - 2.0 * r_dec_down
        μ2 = r_dec + r_up - r_down + 2.0 * r_dec_up
        μ3 = -r_up + r_down - r_dec_up + r_dec_down

        σ21 = r_dec + r_up + r_down + 4.0 * r_dec_down
        σ22 = r_dec + r_up + r_down + 4.0 * r_dec_up
        σ23 = r_up + r_down + r_dec_up + r_dec_down

        g1 = max(1.0, x1)
        g2 = max(1.0, x2)
        g3 = max(1.0, x3)

        τ1_drift = abs(μ1) > 0.0 ? ε * g1 / abs(μ1) : Inf
        τ2_drift = abs(μ2) > 0.0 ? ε * g2 / abs(μ2) : Inf
        τ3_drift = abs(μ3) > 0.0 ? ε * g3 / abs(μ3) : Inf

        τ1_diff = σ21 > 0.0 ? ε^2 * g1^2 / σ21 : Inf
        τ2_diff = σ22 > 0.0 ? ε^2 * g2^2 / σ22 : Inf
        τ3_diff = σ23 > 0.0 ? ε^2 * g3^2 / σ23 : Inf

        τ_leap = min(
            τ1_drift,
            τ2_drift,
            τ3_drift,
            τ1_diff,
            τ2_diff,
            τ3_diff,
        )

        # Never leap past the next requested sample time. This ensures that
        # the sampled trajectory remains piecewise constant on the save grid.
        τ_to_save = Float64(saveat[isave]) - t
        τ_leap = min(τ_leap, τ_to_save)

        do_leap = isfinite(τ_leap) &&
                  τ_leap > 0.0 &&
                  r_total * τ_leap >= min_expected_events

        if do_leap
            τ = τ_leap
            accepted = false

            for _ in 1:max_rejections
                K_dec = rand(rng, Poisson(r_dec * τ))
                K_up = rand(rng, Poisson(r_up * τ))
                K_down = rand(rng, Poisson(r_down * τ))
                K_dec_down = rand(rng, Poisson(r_dec_down * τ))
                K_dec_up = rand(rng, Poisson(r_dec_up * τ))

                S_new = S + K_up - K_down - K_dec_down + K_dec_up
                M_new = M - K_dec - K_dec_down - K_dec_up

                if 0.0 <= S_new <= halfN && -S_new <= M_new <= S_new
                    S = S_new
                    M = M_new
                    t += τ
                    accepted = true
                    break
                end

                τ *= 0.5

                if r_total * τ < min_expected_events
                    break
                end
            end

            if accepted
                value = observable(S, M, N, Γ, ξ)

                while isave <= length(saveat) && saveat[isave] <= t
                    @inbounds out[isave] = value
                    isave += 1
                end

                continue
            end
        end

        # Exact SSA fallback.
        t_next = t - log(rand(rng)) / r_total
        value = observable(S, M, N, Γ, ξ)

        while isave <= length(saveat) && saveat[isave] < t_next
            @inbounds out[isave] = value
            isave += 1
        end

        isave > length(saveat) && break

        r_val = rand(rng) * r_total

        if r_val < r_dec
            M -= 1.0
        elseif r_val < r_dec + r_up
            S += 1.0
        elseif r_val < r_dec + r_up + r_down
            S -= 1.0
        elseif r_val < r_dec + r_up + r_down + r_dec_up
            S += 1.0
            M -= 1.0
        else
            S -= 1.0
            M -= 1.0
        end

        t = t_next
    end

    return out
end

function simulate_ensemble_tau(
    N::Int,
    Γ::Float64,
    ξ::Float64,
    γ::Float64,
    t_max::Float64,
    observable,
    N_traj::Int;
    M0::Int = N ÷ 2,
    S0::Int = N ÷ 2,
    nsave::Int = 2_000,
    ε::Float64 = 0.03,
    min_expected_events::Float64 = 20.0,
    max_rejections::Int = 20,
    rng = Xoshiro(1234),
)
    saveat = collect(range(0.0, t_max, length = nsave))
    avg_obs = zeros(Float64, nsave)

    @inbounds for _ in 1:N_traj
        avg_obs .+= run_trajectory_sampled_adaptive_tau(
            N,
            Γ,
            ξ,
            γ,
            saveat,
            observable;
            M0 = M0,
            S0 = S0,
            ε = ε,
            min_expected_events = min_expected_events,
            max_rejections = max_rejections,
            rng = rng,
        )
    end

    avg_obs ./= N_traj

    return saveat, avg_obs
end

function mc_data_tau(
    N,
    params,
    N_traj;
    time_factor = 3.0,
    nsave = 2000,
    ε = 0.03,
    min_expected_events = 20.0,
    max_rejections = 20,
)
    Γ = params["global_decay"]
    ξ = params["local_dephasing"]
    γ = params["local_decay"]

    tmax = time_factor * log(N) / N
    saveat, intens_mc = simulate_ensemble_tau(
        N,
        Float64(Γ),
        Float64(ξ),
        Float64(γ),
        tmax,
        (S, M, Nobs, Γ, ξ) -> (S + M) * (S - M + 1),
        N_traj;
        nsave = nsave,
        ε = Float64(ε),
        min_expected_events = Float64(min_expected_events),
        max_rejections = max_rejections,
    )

    i_peak = argmax(intens_mc)
    return intens_mc[i_peak], saveat[i_peak]
end

function simulate_ensemble_sampled(
    N,
    Γ,
    ξ,
    γ,
    t_max::Float64,
    observable,
    N_traj;
    nsave = 10_000,
    M0 = N ÷ 2,
    S0 = N ÷ 2,
    rng = Xoshiro(1234),
)
    saveat = collect(range(0.0, t_max, length = nsave))
    avg_obs = zeros(Float64, nsave)

    @inbounds for _ in 1:N_traj
        avg_obs .+= run_trajectory_sampled(
            N,
            Γ,
            ξ,
            γ,
            saveat,
            observable;
            M0 = M0,
            S0 = S0,
            rng = rng,
        )
    end

    avg_obs ./= N_traj

    return saveat, avg_obs
end

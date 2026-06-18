#!/usr/bin/env julia

include(joinpath(@__DIR__, "MonteCarloMF.jl"))

using DifferentialEquations
using JLD2
using Printf
using Random
using Statistics

if !haskey(ENV, "MPLCONFIGDIR")
    mpl_config_dir = joinpath(tempdir(), "matplotlib")
    mkpath(mpl_config_dir)
    ENV["MPLCONFIGDIR"] = mpl_config_dir
end
if !haskey(ENV, "MPLBACKEND")
    ENV["MPLBACKEND"] = "Agg"
end

using PyPlot

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))

state_observable(S, M, N, Gamma, xi) = (S = S, M = M)

function dicke_Gn(S, M, n::Int)
    n >= 1 || error("correlation order n must be >= 1")

    out = 1.0
    for j in 0:(n - 1)
        out *= Float64(S + M - j) * Float64(S - M + 1 + j)
    end
    return out
end

function safe_divide(num::AbstractVector, den::AbstractVector; eps = 1e-14)
    out = Vector{Float64}(undef, length(num))
    for i in eachindex(num, den)
        out[i] = abs(den[i]) > eps ? num[i] / den[i] : NaN
    end
    return out
end

function normalize_moments(moment_matrix, orders)
    idx1 = findfirst(==(1), orders)
    idx1 === nothing && error("orders must contain n = 1")

    G1 = moment_matrix[:, idx1]
    normalized = fill(NaN, size(moment_matrix))
    for (j, n) in enumerate(orders)
        normalized[:, j] = safe_divide(moment_matrix[:, j], G1 .^ n)
    end
    return normalized
end

function simulate_full_jump_reference(
    N::Int,
    Gamma::Float64,
    xi::Float64,
    gamma::Float64,
    t_grid,
    orders,
    n_traj::Int;
    sizehint::Int = 0,
    store_histories::Bool = true,
)
    sums = zeros(Float64, length(t_grid), length(orders))
    histories = Any[]
    store_histories && sizehint!(histories, n_traj)

    for traj in 1:n_traj
        ts, states = run_trajectory(
            N, Gamma, xi, gamma, Float64(last(t_grid)), state_observable;
            sizehint = sizehint,
        )

        if store_histories
            push!(
                histories,
                (
                    t = collect(ts),
                    S = [state.S for state in states],
                    M = [state.M for state in states],
                ),
            )
        end

        idx = 1
        for (k, t) in enumerate(t_grid)
            while idx < length(ts) && ts[idx + 1] <= t
                idx += 1
            end

            S = states[idx].S
            M = states[idx].M
            for (j, n) in enumerate(orders)
                sums[k, j] += dicke_Gn(S, M, n)
            end
        end

        if traj % max(1, n_traj ÷ 10) == 0
            @printf("  full MC: %d/%d trajectories\n", traj, n_traj)
        end
    end

    moments = sums ./ n_traj
    return (
        G = moments,
        g = normalize_moments(moments, orders),
        histories = histories,
    )
end

function run_seed_trajectory(
    N::Int,
    Gamma::Float64,
    xi::Float64,
    gamma::Float64,
    t_max::Float64,
    c::Float64;
    M0::Int = N ÷ 2,
    S0::Int = N ÷ 2,
)
    S::Int = S0
    M::Int = M0
    t = 0.0
    threshold = c * Float64(N)
    G1 = dicke_Gn(S, M, 1)

    if G1 >= threshold
        return (reached = true, t = t, S = S, M = M, G1 = G1)
    end

    while t < t_max
        r_dec = Gamma * A_JM_minus2(S, M) + gamma * P_JM_minus_02(S, M, N)
        r_up = xi * P_JM_z_plus2(S, M, N)
        r_down = xi * P_JM_z_minus2(S, M, N)
        r_dec_down = gamma * P_JM_minus_minus2(S, M, N)
        r_dec_up = gamma * P_JM_minus_plus2(S, M, N)

        r_total = r_dec + r_up + r_down + r_dec_up + r_dec_down
        if r_total <= 1e-10
            return (reached = false, t = Inf, S = S, M = M, G1 = G1)
        end

        tau = -log(rand()) / r_total
        t_next = t + tau

        r_val = rand() * r_total
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
        G1 = dicke_Gn(S, M, 1)
        if t <= t_max && G1 >= threshold
            return (reached = true, t = t, S = S, M = M, G1 = G1)
        end
    end

    return (reached = false, t = Inf, S = S, M = M, G1 = G1)
end

function simulate_seed_times(
    N::Int,
    Gamma::Float64,
    xi::Float64,
    gamma::Float64,
    t_max::Float64,
    c::Float64,
    n_seed::Int,
)
    seeds = Vector{Any}(undef, n_seed)
    for traj in 1:n_seed
        seeds[traj] = run_seed_trajectory(N, Gamma, xi, gamma, t_max, c)
        if traj % max(1, n_seed ÷ 10) == 0
            @printf("  seed MC c = %.4g: %d/%d trajectories\n", c, traj, n_seed)
        end
    end
    return seeds
end

function solve_seed_meanfield(
    N::Int,
    Gamma::Float64,
    xi::Float64,
    gamma::Float64,
    t_max::Float64,
    I0::Float64,
    m0::Float64;
    reltol = 1e-9,
    abstol = 1e-11,
)
    p = [Gamma, xi, N, gamma]
    prob = ODEProblem(meanfield_reduced!, [I0, m0], (0.0, t_max), p)
    return solve(
        prob,
        AutoTsit5(Rosenbrock23()),
        reltol = reltol,
        abstol = abstol,
    )
end

function reconstruct_from_seed_times(
    N::Int,
    Gamma::Float64,
    xi::Float64,
    gamma::Float64,
    t_grid,
    seeds,
    c::Float64,
    orders;
    continuation::Symbol = :seed_state,
    reltol = 1e-9,
    abstol = 1e-11,
)
    sums = zeros(Float64, length(t_grid), length(orders))
    cache = Dict{Any, Any}()
    t_max = Float64(last(t_grid) - first(t_grid))

    for seed in seeds
        seed.reached || continue

        if continuation == :seed_state
            key = (seed.S, seed.M)
            I0 = seed.G1 / Float64(N)^2
            m0 = Float64(seed.M) / Float64(N)
        elseif continuation == :fixed
            key = :fixed
            I0 = c / Float64(N)
            m0 = 0.5
        else
            error("continuation must be :seed_state or :fixed")
        end

        sol = get!(cache, key) do
            solve_seed_meanfield(
                N, Gamma, xi, gamma, t_max, Float64(I0), Float64(m0);
                reltol = reltol,
                abstol = abstol,
            )
        end

        for (i, t) in enumerate(t_grid)
            t_rel = t - seed.t
            if t_rel < 0.0
                continue
            end

            Ifrak = max(Float64(sol(t_rel)[1]), 0.0)
            G1 = Float64(N)^2 * Ifrak
            for (j, n) in enumerate(orders)
                sums[i, j] += G1^n
            end
        end
    end

    moments = sums ./ length(seeds)
    return (
        G1_powers = moments,
        g_proxy = normalize_moments(moments, orders),
        continuation = continuation,
        n_unique_meanfield_paths = length(cache),
    )
end

function seed_time_summary(N, Gamma, seeds)
    reached = [seed.reached for seed in seeds]
    tau_reached = [N * Gamma * seed.t for seed in seeds if seed.reached]

    return (
        reached_fraction = mean(reached),
        tau0_mean = isempty(tau_reached) ? NaN : mean(tau_reached),
        tau0_std = length(tau_reached) <= 1 ? NaN : std(tau_reached),
        tau0_min = isempty(tau_reached) ? NaN : minimum(tau_reached),
        tau0_max = isempty(tau_reached) ? NaN : maximum(tau_reached),
    )
end

function comparison_summary(curve, orders; bulk_I_min = 5e-4)
    idx1 = findfirst(==(1), orders)
    full_I = curve.full_G[:, idx1] ./ curve.N^2
    seed_I = curve.seed_G1_powers[:, idx1] ./ curve.N^2
    intensity_mask = isfinite.(full_I) .& isfinite.(seed_I) .&
                     (full_I .> bulk_I_min) .& (seed_I .> bulk_I_min)

    if any(intensity_mask)
        I_diff = abs.(full_I[intensity_mask] .- seed_I[intensity_mask])
        intensity_mean_abs_error = mean(I_diff)
        intensity_max_abs_error = maximum(I_diff)
    else
        intensity_mean_abs_error = NaN
        intensity_max_abs_error = NaN
    end

    g_summaries = Any[]
    for (j, n) in enumerate(orders)
        n == 1 && continue

        g_full = curve.full_g[:, j]
        g_seed = curve.seed_g_proxy[:, j]
        mask = isfinite.(g_full) .& isfinite.(g_seed) .&
               (full_I .> bulk_I_min) .& (seed_I .> bulk_I_min)

        if any(mask)
            diff = abs.(g_full[mask] .- g_seed[mask])
            push!(
                g_summaries,
                (
                    n = n,
                    mean_abs_error = mean(diff),
                    max_abs_error = maximum(diff),
                ),
            )
        else
            push!(
                g_summaries,
                (
                    n = n,
                    mean_abs_error = NaN,
                    max_abs_error = NaN,
                ),
            )
        end
    end

    peak_full = argmax(full_I)
    peak_seed = argmax(seed_I)
    return (
        intensity_mean_abs_error = intensity_mean_abs_error,
        intensity_max_abs_error = intensity_max_abs_error,
        tau_peak_full = curve.tau[peak_full],
        tau_peak_seed = curve.tau[peak_seed],
        I_peak_full = full_I[peak_full],
        I_peak_seed = seed_I[peak_seed],
        g = g_summaries,
    )
end

function finite_mask(y)
    return isfinite.(y)
end

function plot_seed_curve(data, curve; outfile)
    mkpath(dirname(outfile))

    idx1 = findfirst(==(1), data.orders)
    full_I = curve.full_G[:, idx1] ./ data.N^2
    seed_I = curve.seed_G1_powers[:, idx1] ./ data.N^2
    tau0 = [data.N * data.Gamma * seed.t for seed in curve.seeds if seed.reached]
    orders_to_plot = [n for n in data.orders if n > 1]

    fig, axes = subplots(4, 1, figsize = (7.2, 9.0), sharex = false)
    cmap = get_cmap("viridis")
    colors = length(orders_to_plot) == 1 ?
             [cmap(0.55)] :
             [cmap(x) for x in range(0.15, 0.85, length = length(orders_to_plot))]

    bins = isempty(tau0) ? 10 : min(60, max(8, ceil(Int, sqrt(length(tau0)))))
    axes[1].hist(tau0, bins = bins, color = "0.25", alpha = 0.82)
    axes[1].set_ylabel("count")
    axes[1].set_title(@sprintf(
        "seed threshold c = %.4g, reached %.1f%%",
        curve.c,
        100 * curve.seed_summary.reached_fraction,
    ))
    axes[1].grid(true, alpha = 0.25)

    axes[2].plot(curve.tau, full_I, color = "black", lw = 1.8, label = "full MC")
    positive_seed = seed_I .> 0
    axes[2].plot(curve.tau[positive_seed], seed_I[positive_seed], color = "tab:blue", lw = 1.8, ls = "--", label = "seed + MF")
    axes[2].set_ylabel("<G1>/N^2")
    axes[2].set_yscale("log")
    axes[2].legend(frameon = false)
    axes[2].grid(true, alpha = 0.25)

    for (color, n) in zip(colors, orders_to_plot)
        j = findfirst(==(n), data.orders)
        full_mask = finite_mask(curve.full_g[:, j]) .& (full_I .> data.plot_I_min)
        seed_mask = finite_mask(curve.seed_g_proxy[:, j]) .& (seed_I .> data.plot_I_min)
        axes[3].plot(
            curve.tau[full_mask],
            curve.full_g[full_mask, j],
            color = color,
            lw = 1.7,
            label = @sprintf("full n=%d", n),
        )
        axes[3].plot(
            curve.tau[seed_mask],
            curve.seed_g_proxy[seed_mask, j],
            color = color,
            lw = 1.7,
            ls = "--",
            label = @sprintf("seed n=%d", n),
        )

        diff_mask = full_mask .& seed_mask
        axes[4].plot(
            curve.tau[diff_mask],
            curve.full_g[diff_mask, j] .- curve.seed_g_proxy[diff_mask, j],
            color = color,
            lw = 1.5,
            label = @sprintf("n=%d", n),
        )
    end

    axes[3].set_ylabel("normalized correlation/proxy")
    axes[3].legend(frameon = false, fontsize = 8, ncol = 2)
    axes[3].grid(true, alpha = 0.25)

    axes[4].axhline(0.0, color = "0.35", lw = 0.9)
    axes[4].set_xlabel("tau = N Gamma t")
    axes[4].set_ylabel("full - seed")
    axes[4].legend(frameon = false, fontsize = 8)
    axes[4].grid(true, alpha = 0.25)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches = "tight")
    close(fig)
    return outfile
end

safe_label(x) = replace(replace(@sprintf("%.6g", x), "." => "p"), "-" => "m")

function compare_seed_time_mf_trajectories(;
    N = 5000,
    Gamma = 1.0,
    g_xi = 0.5,
    gamma = 0.0,
    c_values = [1.0, 2.0, 5.0, 10.0],
    orders = [1, 2, 3],
    n_full = 300,
    n_seed = n_full,
    t_fac = 12.0,
    nt = 1500,
    seed = 4321,
    continuation = :seed_state,
    store_full_histories = true,
    bulk_I_min = 5.0 / N,
    plot_I_min = 1.0 / N^2,
    outfile = joinpath(ROOT_DIR, "plot_data", "seed_time_mf_comparison.jld2"),
    plot_dir = joinpath(ROOT_DIR, "data", "seed_time_mf"),
)
    Random.seed!(seed)
    mkpath(dirname(outfile))
    mkpath(plot_dir)

    orders = sort(unique(Int.(vcat(1, orders))))
    c_values = Float64.(c_values)
    Gamma = Float64(Gamma)
    xi = Float64(g_xi) * N * Gamma
    gamma = Float64(gamma)

    t_max = Float64(t_fac) * log(N) / (N * Gamma)
    t_grid = collect(range(0.0, t_max, length = nt))
    tau_grid = N .* Gamma .* t_grid

    @printf(
        "Full jump MC reference: N = %d, g_xi = %.4g, gamma = %.4g, trajectories = %d\n",
        N,
        g_xi,
        gamma,
        n_full,
    )
    full = simulate_full_jump_reference(
        N, Gamma, xi, gamma, t_grid, orders, n_full;
        sizehint = max(100, Int(ceil(2 * N))),
        store_histories = store_full_histories,
    )

    curves = Vector{Any}(undef, length(c_values))
    plotfiles = Vector{String}(undef, length(c_values))

    for (i, c) in enumerate(c_values)
        @printf("\nSeed threshold c = %.4g\n", c)
        seeds = simulate_seed_times(N, Gamma, xi, gamma, t_max, c, n_seed)
        seed_reconstruction = reconstruct_from_seed_times(
            N, Gamma, xi, gamma, t_grid, seeds, c, orders;
            continuation = continuation,
        )

        curve_without_summary = (
            N = N,
            c = c,
            t = t_grid,
            tau = tau_grid,
            seeds = seeds,
            seed_summary = seed_time_summary(N, Gamma, seeds),
            full_G = full.G,
            full_g = full.g,
            seed_G1_powers = seed_reconstruction.G1_powers,
            seed_g_proxy = seed_reconstruction.g_proxy,
            continuation = seed_reconstruction.continuation,
            n_unique_meanfield_paths = seed_reconstruction.n_unique_meanfield_paths,
        )

        curves[i] = merge(
            curve_without_summary,
            (summary = comparison_summary(curve_without_summary, orders; bulk_I_min = bulk_I_min),),
        )

        plotfiles[i] = plot_seed_curve(
            (
                N = N,
                Gamma = Gamma,
                orders = orders,
                plot_I_min = plot_I_min,
            ),
            curves[i];
            outfile = joinpath(plot_dir, "seed_time_mf_c$(safe_label(c)).pdf"),
        )

        s = curves[i].summary
        @printf(
            "  reached %.1f%%, tau0 mean %.4g, tau peak full %.4g, seed %.4g, peak I full %.4g, seed %.4g\n",
            100 * curves[i].seed_summary.reached_fraction,
            curves[i].seed_summary.tau0_mean,
            s.tau_peak_full,
            s.tau_peak_seed,
            s.I_peak_full,
            s.I_peak_seed,
        )
    end

    data = (
        N = N,
        Gamma = Gamma,
        g_xi = Float64(g_xi),
        xi = xi,
        gamma = gamma,
        c_values = c_values,
        orders = orders,
        n_full = n_full,
        n_seed = n_seed,
        t_fac = Float64(t_fac),
        nt = nt,
        seed = seed,
        continuation = continuation,
        store_full_histories = store_full_histories,
        t_max = t_max,
        t = t_grid,
        tau = tau_grid,
        bulk_I_min = bulk_I_min,
        plot_I_min = plot_I_min,
        full_histories = full.histories,
        curves = curves,
        outfile = outfile,
        plotfiles = plotfiles,
    )

    jldsave(outfile; data...)
    @printf("\nSaved data: %s\n", outfile)
    for file in plotfiles
        @printf("Saved plot: %s\n", file)
    end

    return data
end

function parse_cli_args(args)
    opts = Dict{String, String}()
    for arg in args
        startswith(arg, "--") || continue
        parts = split(arg[3:end], "=", limit = 2)
        if length(parts) == 2
            opts[parts[1]] = parts[2]
        end
    end
    return opts
end

parse_float_list(x) = [parse(Float64, strip(part)) for part in split(x, ",") if !isempty(strip(part))]
parse_int_list(x) = [parse(Int, strip(part)) for part in split(x, ",") if !isempty(strip(part))]
parse_bool(x) = lowercase(strip(x)) in ["1", "true", "yes", "y"]

function main()
    opts = parse_cli_args(ARGS)
    compare_seed_time_mf_trajectories(
        N = parse(Int, get(opts, "N", "5000")),
        Gamma = parse(Float64, get(opts, "Gamma", "1.0")),
        g_xi = parse(Float64, get(opts, "g-xi", "0.5")),
        gamma = parse(Float64, get(opts, "gamma", "0.0")),
        c_values = parse_float_list(get(opts, "c", "1,2,5,10")),
        orders = parse_int_list(get(opts, "orders", "1,2,3")),
        n_full = parse(Int, get(opts, "n-full", "300")),
        n_seed = parse(Int, get(opts, "n-seed", get(opts, "n-full", "300"))),
        t_fac = parse(Float64, get(opts, "t-fac", "12.0")),
        nt = parse(Int, get(opts, "nt", "1500")),
        seed = parse(Int, get(opts, "seed", "4321")),
        continuation = Symbol(get(opts, "continuation", "seed_state")),
        store_full_histories = parse_bool(get(opts, "store-full-histories", "true")),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

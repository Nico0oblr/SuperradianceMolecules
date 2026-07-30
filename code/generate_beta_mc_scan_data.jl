include("MonteCarloMF.jl")

using Base.Threads
using JLD2

function log_spaced_ints(lo, hi, n)
    values = Int.(round.(10 .^ range(log10(lo), log10(hi), length = n)))
    return unique(values)
end

function scan_range(lo, hi, n)
    n == 1 && return [Float64(lo)]
    return range(Float64(lo), Float64(hi), length = n)
end

function generate_beta_mc_scan_data(;
    outfile = "../plot_data/beta_mc_scan_data.jld2",
    n_xi = 50,
    n_gamma = 50,
    n_N = 10,
    g_xis = scan_range(0.0, 1.1, n_xi),
    gbars_gamma = scan_range(0.0, 0.5, n_gamma),
    Ns = log_spaced_ints(1e4, 545559478, n_N),
    N_traj = 500,
    time_factor = 8.0,
    nsave = 2000,
    ε = 0.03,
    min_expected_events = 20.0,
    max_rejections = 20,
    save_progress = false,
)
    Γ = 1.0

    g_xis = collect(Float64.(g_xis))
    gbars_gamma = collect(Float64.(gbars_gamma))
    Ns = collect(Int.(Ns))

    mc_peak_intens = zeros(Float64, length(g_xis), length(gbars_gamma), length(Ns))
    mc_peak_ifrak = zeros(Float64, length(g_xis), length(gbars_gamma), length(Ns))
    mc_peak_t = zeros(Float64, length(g_xis), length(gbars_gamma), length(Ns))
    xis = zeros(Float64, length(g_xis), length(gbars_gamma), length(Ns))
    gammas = zeros(Float64, length(g_xis), length(gbars_gamma), length(Ns))

    jobs = collect(CartesianIndices((length(g_xis), length(gbars_gamma), length(Ns))))
    progress_lock = ReentrantLock()
    completed = Ref(0)

    @threads for job in jobs
        i, j, k = Tuple(job)
        g_xi = g_xis[i]
        gbar_gamma = gbars_gamma[j]
        N = Ns[k]

        ξ = g_xi * N * Γ
        γ = gbar_gamma * N * Γ / log(N)

        params = Dict(
            "global_decay" => Γ,
            "global_pump" => 0.0,
            "local_pump" => 0.0,
            "local_dephasing" => ξ,
            "local_decay" => γ,
        )

        I_peak, t_peak = mc_data_tau(
            N,
            params,
            N_traj;
            time_factor = time_factor,
            nsave = nsave,
            ε = ε,
            min_expected_events = min_expected_events,
            max_rejections = max_rejections,
        )

        lock(progress_lock) do
            xis[i, j, k] = ξ
            gammas[i, j, k] = γ
            mc_peak_intens[i, j, k] = I_peak
            mc_peak_ifrak[i, j, k] = I_peak / N^2
            mc_peak_t[i, j, k] = t_peak

            completed[] += 1
            println(
                "Beta MC scan: g_xi=$(round(g_xi, digits=4)) [$i/$(length(g_xis))], ",
                "gbar_gamma=$(round(gbar_gamma, digits=4)) [$j/$(length(gbars_gamma))], ",
                "N=$N [$k/$(length(Ns))], completed $(completed[])/$(length(jobs))",
            )

            if save_progress
                jldsave(outfile;
                    g_xis = g_xis,
                    gbars_gamma = gbars_gamma,
                    Ns = Ns,
                    N_traj = N_traj,
                    method = "tau_leaping",
                    time_factor = time_factor,
                    nsave = nsave,
                    ε = ε,
                    min_expected_events = min_expected_events,
                    max_rejections = max_rejections,
                    mc_peak_intens = mc_peak_intens,
                    mc_peak_ifrak = mc_peak_ifrak,
                    mc_peak_t = mc_peak_t,
                    xis = xis,
                    gammas = gammas,
                )
            end
        end
    end

    jldsave(outfile;
        g_xis = g_xis,
        gbars_gamma = gbars_gamma,
        Ns = Ns,
        N_traj = N_traj,
        method = "tau_leaping",
        time_factor = time_factor,
        nsave = nsave,
        ε = ε,
        min_expected_events = min_expected_events,
        max_rejections = max_rejections,
        mc_peak_intens = mc_peak_intens,
        mc_peak_ifrak = mc_peak_ifrak,
        mc_peak_t = mc_peak_t,
        xis = xis,
        gammas = gammas,
    )

    return (
        g_xis = g_xis,
        gbars_gamma = gbars_gamma,
        Ns = Ns,
        N_traj = N_traj,
        method = "tau_leaping",
        time_factor = time_factor,
        nsave = nsave,
        ε = ε,
        min_expected_events = min_expected_events,
        max_rejections = max_rejections,
        mc_peak_intens = mc_peak_intens,
        mc_peak_ifrak = mc_peak_ifrak,
        mc_peak_t = mc_peak_t,
        xis = xis,
        gammas = gammas,
    )
end

function main()
    generate_beta_mc_scan_data()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

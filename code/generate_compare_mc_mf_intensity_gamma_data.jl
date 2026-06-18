include("PermBasis.jl")
include("MonteCarloMF.jl")

using DifferentialEquations
using JLD2

function generate_compare_mc_mf_intensity_gamma_data(;
    N = 30000,
    t_fac = 20,
    N_traj = 1000,
    g_xi = 0.5,
    gbars_gamma = [0.0, 0.2, 0.5, 0.8],
    outfile = "../plot_data/fig_compare_mc_mf_intensity_gamma.jld2",
)
    Γ = 1.0
    ξ = g_xi * N * Γ
    tmax = t_fac * log(N) / N

    curves = Vector{NamedTuple}(undef, length(gbars_gamma))

    for (i, gbar_gamma) in enumerate(gbars_gamma)
        γ = gbar_gamma * N * Γ / log(N)

        ts_mc, intens_mc = simulate_ensemble(
            N, Γ, ξ, γ, tmax,
            (S, M, N, Γ, ξ) -> [(S + M) * (S - M + 1)],
            N_traj
        )
        I_mc = element(intens_mc, 1) ./ N^2

        p = [Γ, ξ, N, γ]
        u0 = [1 / N, 1 / 2]
        prob = ODEProblem(meanfield_reduced!, u0, (0.0, tmax), p)
        sol_mf = solve(
            prob,
            AutoTsit5(Rosenbrock23()),
            reltol = 1e-8,
            abstol = 1e-8,
            saveat = range(0.0, tmax, 2000)
        )

        I_mf = [u[1] for u in sol_mf.u]

        curves[i] = (
            gbar_gamma = Float64(gbar_gamma),
            ts_mc = collect(ts_mc),
            I_mc = collect(I_mc),
            ts_mf = collect(sol_mf.t),
            I_mf = collect(I_mf),
        )
    end

    data = (
        N = N,
        t_fac = t_fac,
        N_traj = N_traj,
        g_xi = g_xi,
        gbars_gamma = collect(Float64.(gbars_gamma)),
        tmax = tmax,
        curves = curves,
    )

    jldsave(outfile; data...)
    return data
end

function main()
    generate_compare_mc_mf_intensity_gamma_data()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
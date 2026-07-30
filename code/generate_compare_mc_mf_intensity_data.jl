include("MonteCarloMF.jl")

using Base.Threads
using DifferentialEquations
using JLD2

function generate_compare_mc_mf_intensity_data(;
    N = 1000000,
    t_fac = 20,
    N_traj = 1000,
    rs = [0.0, 0.2, 0.4, 0.7, 0.9, 1.05],
    outfile = "../plot_data/fig_compare_mc_mf_intensity.jld2",
)
    tmax = t_fac * log(N) / N

    curves = Vector{NamedTuple}(undef, length(rs))

    @threads for i in eachindex(rs)
        @show rs[i]
        r = rs[i]
        ts_mc, intens_mc = @time simulate_ensemble_adaptive(N, 1.0, r * N, 0.0, tmax, (S, M, N, Γ, ξ) -> (S + M) * (S - M + 1), N_traj; nsave = 2000)

        I_mc = intens_mc ./ N^2

        p = [1.0, r * N, N, 0.0]
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
            r = Float64(r),
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
        rs = collect(Float64.(rs)),
        tmax = tmax,
        curves = curves,
    )

    jldsave(outfile; data...)
    return data
end

function main()
    generate_compare_mc_mf_intensity_data()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

include("MonteCarloMF.jl")

using DifferentialEquations
using JLD2

function generate_fig2b_gamma_data(;
    N = 10000,
    time_factor = 200,
    N_traj = 400,
    g_xi = 0.0,
    gbars_gamma = [0.0, 0.2, 0.5, 0.8],
    outfile = "../plot_data/fig_trajectories_gamma.jld2",
)
    Γ = 1.0
    ξ = g_xi * N * Γ
    tmax = time_factor * log(N) / N

    curves = Vector{NamedTuple}(undef, length(gbars_gamma))

    for (i, gbar_gamma) in enumerate(gbars_gamma)
        γ = Float64(gbar_gamma) * N * Γ / log(N)
        println("Fig. 4a: gbar_gamma = $(gbar_gamma), simulating $(N_traj) trajectories at N = $(N)")

        times_mc, sol_mc = simulate_ensemble(
            N,
            Γ,
            ξ,
            γ,
            tmax,
            (S, M, N, Γ, ξ) -> [S, M],
            N_traj
        )
        S_mc = collect(element(sol_mc, 1) ./ N)
        M_mc = collect(element(sol_mc, 2) ./ N)

        p = [Γ, ξ, N, γ]
        u0 = [1 / N, 1 / 2]
        prob = ODEProblem(meanfield_reduced!, u0, (0.0, tmax), p)
        mfsol = solve(prob, AutoTsit5(Rosenbrock23()), reltol = 1e-8, abstol = 1e-8)

        M_mf = [u[2] for u in mfsol.u]
        I_mf = [u[1] for u in mfsol.u]
        S_mf = sqrt.(I_mf .+ M_mf .^ 2)

        curves[i] = (
            gbar_gamma = Float64(gbar_gamma),
            gamma = γ,
            times_mc = collect(times_mc),
            S_mc = S_mc,
            M_mc = M_mc,
            times_mf = collect(mfsol.t),
            S_mf = S_mf,
            M_mf = M_mf,
        )
    end

    Sgrid = collect(range(0.0, 0.5, length = 400))
    upper_boundary = copy(Sgrid)
    lower_boundary = -Sgrid
    vertical_boundary_x = [0.5, 0.5]
    vertical_boundary_y = [-0.5, 0.5]

    data = (
        N = N,
        time_factor = time_factor,
        N_traj = N_traj,
        g_xi = g_xi,
        gbars_gamma = collect(Float64.(gbars_gamma)),
        global_decay = Γ,
        local_dephasing = ξ,
        tmax = tmax,
        curves = curves,
        Sgrid = Sgrid,
        upper_boundary = upper_boundary,
        lower_boundary = lower_boundary,
        vertical_boundary_x = vertical_boundary_x,
        vertical_boundary_y = vertical_boundary_y,
    )

    jldsave(outfile; data = data)
    return data
end

function main()
    generate_fig2b_gamma_data()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

include("PermBasis.jl")
include("MonteCarloMF.jl")

using DifferentialEquations
using JLD2

function generate_fig2a_data(;
    N = 10000,
    time_factor = 200,
    N_traj = 400,
    g_xis = [0.2, 0.4, 0.6],
    outfile = "../plot_data/fig2a_data.jld2",
)
    Γ = 1.0
    γ = 0.0

    tmax = time_factor * log(N) / N

    curves = Vector{NamedTuple}(undef, length(g_xis))

    for (i, g_xi) in enumerate(g_xis)
        ξ = Float64(g_xi) * N * Γ
        println("Fig. 2a: g_xi = $(g_xi), simulating $(N_traj) trajectories at N = $(N)")

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
            g_xi = Float64(g_xi),
            xi = ξ,
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

    data = Dict(
        "N" => N,
        "time_factor" => time_factor,
        "N_traj" => N_traj,
        "g_xis" => collect(Float64.(g_xis)),
        "global_decay" => Γ,
        "local_decay" => γ,
        "tmax" => tmax,
        "curves" => curves,
        "Sgrid" => Sgrid,
        "upper_boundary" => upper_boundary,
        "lower_boundary" => lower_boundary,
        "vertical_boundary_x" => vertical_boundary_x,
        "vertical_boundary_y" => vertical_boundary_y,
    )

    jldsave(outfile; data = data)
    return data
end

function main()
    generate_fig2a_data()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

include("PermBasis.jl")
include("MonteCarloMF.jl")

using DifferentialEquations
using JLD2

function generate_fig2b_gamma_data(;
    N = 10000,
    time_factor = 200,
    N_traj = 400,
    n_bg = 10,
    g_xi = 0.0,
    gbar_gamma = 0.2,
    outfile = "../plot_data/fig_trajectories_gamma.jld2",
)
    Γ = 1.0
    ξ = g_xi * N * Γ
    γ = gbar_gamma * N * Γ / log(N)

    params = Dict(
        "global_decay" => Γ,
        "global_pump" => 0.0,
        "local_pump" => 0.0,
        "local_dephasing" => ξ,
        "local_decay" => γ,
    )

    tmax = time_factor * log(N) / N

    bg_trajectories = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, n_bg)
    for j in 1:n_bg
        times, sol = simulate_ensemble(
            N,
            params["global_decay"],
            params["local_dephasing"],
            params["local_decay"],
            tmax,
            (S, M, N, Γ, ξ) -> [S, M],
            1
        )
        bg_trajectories[j] = (
            collect(element(sol, 1) ./ N),
            collect(element(sol, 2) ./ N)
        )
    end

    times_mc, sol_mc = simulate_ensemble(
        N,
        params["global_decay"],
        params["local_dephasing"],
        params["local_decay"],
        tmax,
        (S, M, N, Γ, ξ) -> [S, M],
        N_traj
    )
    S_mc = collect(element(sol_mc, 1) ./ N)
    M_mc = collect(element(sol_mc, 2) ./ N)

    p = [params["global_decay"], params["local_dephasing"], N, params["local_decay"]]
    u0 = [1 / N, 1 / 2]
    prob = ODEProblem(meanfield_reduced!, u0, (0.0, tmax), p)
    mfsol = solve(prob, AutoTsit5(Rosenbrock23()), reltol = 1e-8, abstol = 1e-8)

    M_mf = [u[2] for u in mfsol.u]
    I_mf = [u[1] for u in mfsol.u]
    S_mf = sqrt.(I_mf .+ M_mf .^ 2)

    Sgrid = collect(range(0.0, 0.5, length = 400))
    upper_boundary = copy(Sgrid)
    lower_boundary = -Sgrid
    vertical_boundary_x = [0.5, 0.5]
    vertical_boundary_y = [-0.5, 0.5]

    data = (
        N = N,
        time_factor = time_factor,
        N_traj = N_traj,
        n_bg = n_bg,
        g_xi = g_xi,
        gbar_gamma = gbar_gamma,
        tmax = tmax,
        times_mc = collect(times_mc),
        S_mc = S_mc,
        M_mc = M_mc,
        times_mf = collect(mfsol.t),
        S_mf = S_mf,
        M_mf = M_mf,
        bg_trajectories = bg_trajectories,
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
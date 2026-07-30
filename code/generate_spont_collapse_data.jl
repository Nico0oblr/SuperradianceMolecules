include("MonteCarloMF.jl")

using Base.Threads
using JLD2

function generate_spont_collapse_data(;
    outfile = "../plot_data/spont_collapse_data.jld2",
    Ns = [10000, 100000, 1000000],
    N_traj = 500,
    gs = range(0.0, 1.0, 300),
    r = 0.0,
    time_factor = 8.0,
)
    mf_itens = zeros(length(gs), length(Ns))
    mc_intens = zeros(length(gs), length(Ns))
    mf_t = zeros(length(gs), length(Ns))
    mc_t = zeros(length(gs), length(Ns))

    jobs = collect(CartesianIndices((length(gs), length(Ns))))

    @threads for job in jobs
        i, j = Tuple(job)
        g = gs[i]
        Nval = Ns[j]
        @show g Nval

        params = Dict(
            "global_decay" => 1.0,
            "global_pump" => 0.0,
            "local_pump" => 0.0,
            "local_dephasing" => r * Nval,
            "local_decay" => g * Nval / log(Nval),
        )

        result = mc_vs_mf_adaptive(Nval, params, N_traj; time_factor=time_factor)

        mf_itens[i, j] = result.I_mf
        mc_intens[i, j] = result.I_mc
        mf_t[i, j] = result.t_mf
        mc_t[i, j] = result.t_mc
    end

    jldsave(outfile;
        Ns = Ns,
        N_traj = N_traj,
        gs = collect(gs),
        r = r,
        time_factor = time_factor,
        mf_itens = mf_itens,
        mc_intens = mc_intens,
        mf_t = mf_t,
        mc_t = mc_t,
    )
end

function main()
    generate_spont_collapse_data()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

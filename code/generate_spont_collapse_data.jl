include("PermBasis.jl")
include("MonteCarloMF.jl")

using JLD2

function generate_spont_collapse_data(;
    outfile = "../plot_data/spont_collapse_data.jld2",
    Ns = [1000, 10000, 50000],
    N_traj = 500,
    gs = range(0.0, 1.0, 300),
    r = 0.0,
    time_factor = 8.0,
)
    mf_itens = zeros(length(gs), length(Ns))
    mc_intens = zeros(length(gs), length(Ns))
    mf_t = zeros(length(gs), length(Ns))
    mc_t = zeros(length(gs), length(Ns))

    for (i, g) in enumerate(gs)
        for (j, N) in enumerate(Ns)
            params = Dict(
                "global_decay" => 1.0,
                "global_pump" => 0.0,
                "local_pump" => 0.0,
                "local_dephasing" => r * N,
                "local_decay" => g * N / log(N),
            )

            imc, imf, tmc, tmf = mc_vs_mf_data(N, params, N_traj; time_factor=time_factor)

            mf_itens[i, j] = imf
            mc_intens[i, j] = imc
            mf_t[i, j] = tmf
            mc_t[i, j] = tmc
        end
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
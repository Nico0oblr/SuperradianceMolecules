include("PermBasis.jl")
include("MonteCarloMF.jl")

using JLD2

function generate_dephasing_main_scan(;
    Ns = [1000, 10000, 50000],
    N_traj = 500,
    rs = range(0.0, 2.0, 300),
    g = 0.0,
    time_factor = 50.0,
    outfile = "../plot_data/dephasing_main_scan.jld2",
    save_progress = true,
)
    mf_intens = zeros(length(rs), length(Ns))
    mc_intens = zeros(length(rs), length(Ns))
    mf_t = zeros(length(rs), length(Ns))
    mc_t = zeros(length(rs), length(Ns))

    for (i, r) in enumerate(rs)
        println("Main scan: r = $(round(r, digits=4)) [$i/$(length(rs))]")
        for (j, N) in enumerate(Ns)
            println("  N = $N [$j/$(length(Ns))]")

            params = Dict(
                "global_decay" => 1.0,
                "global_pump" => 0.0,
                "local_pump" => 0.0,
                "local_dephasing" => r * N,
                "local_decay" => g * N / log(N),
            )

            imc, imf, tmc, tmf = mc_vs_mf_data(N, params, N_traj; time_factor=time_factor)

            mf_intens[i, j] = imf
            mc_intens[i, j] = imc
            mf_t[i, j] = tmf
            mc_t[i, j] = tmc
        end

        if save_progress
            jldsave(outfile;
                rs = collect(rs),
                Ns = Ns,
                N_traj = N_traj,
                g = g,
                time_factor = time_factor,
                mf_intens = mf_intens,
                mc_intens = mc_intens,
                mf_t = mf_t,
                mc_t = mc_t,
            )
        end
    end

    jldsave(outfile;
        rs = collect(rs),
        Ns = Ns,
        N_traj = N_traj,
        g = g,
        time_factor = time_factor,
        mf_intens = mf_intens,
        mc_intens = mc_intens,
        mf_t = mf_t,
        mc_t = mc_t,
    )
end

function main()
    generate_dephasing_main_scan()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
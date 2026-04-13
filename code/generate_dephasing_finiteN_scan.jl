include("PermBasis.jl")
include("MonteCarloMF.jl")

using JLD2

function generate_dephasing_finiteN_scan(;
    Ns = Int.(round.(10 .^ range(4, 7.0, 50))),
    rs = [0.1, 0.95, 1.05],
    N_traj = 800,
    g = 0.0,
    time_factor = 16.0,
    outfile = "../plot_data/dephasing_finiteN_scan.jld2",
    save_progress = true,
)
    mc_intens = zeros(length(rs), length(Ns))
    mc_t = zeros(length(rs), length(Ns))

    for (i, r) in enumerate(rs)
        println("Finite-N scan: r = $r [$i/$(length(rs))]")
        for (j, N) in enumerate(Ns)
            println("  N = $N [$j/$(length(Ns))]")

            params = Dict(
                "global_decay" => 1.0,
                "global_pump" => 0.0,
                "local_pump" => 0.0,
                "local_dephasing" => r * N,
                "local_decay" => g * N / log(N),
            )

            imc, tmc = mc_data(N, params, N_traj; time_factor = time_factor)

            mc_intens[i, j] = imc
            mc_t[i, j] = tmc
        end

        if save_progress
            jldsave(outfile;
                rs = rs,
                Ns = Ns,
                N_traj = N_traj,
                g = g,
                time_factor = time_factor,
                mc_intens = mc_intens,
                mc_t = mc_t,
            )
        end
    end

    jldsave(outfile;
        rs = rs,
        Ns = Ns,
        N_traj = N_traj,
        g = g,
        time_factor = time_factor,
        mc_intens = mc_intens,
        mc_t = mc_t,
    )

    return (
        rs = rs,
        Ns = Ns,
        N_traj = N_traj,
        g = g,
        time_factor = time_factor,
        mc_intens = mc_intens,
        mc_t = mc_t,
    )
end

function main()
    generate_dephasing_finiteN_scan()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
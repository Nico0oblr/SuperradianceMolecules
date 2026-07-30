include("MonteCarloMF.jl")

using Base.Threads
using JLD2

function generate_dephasing_finiteN_scan(;
    Ns = Int.(round.(10 .^ range(4, 7.0, 50))),
    rs = [0.6, 0.8, 0.95, 1.05, 1.1, 1.15],
    N_traj = 800,
    g = 0.0,
    time_factor = 16.0,
    outfile = "../plot_data/dephasing_finiteN_scan.jld2",
    save_progress = false,
)
    mc_intens = zeros(length(rs), length(Ns))
    mc_t = zeros(length(rs), length(Ns))

    jobs = collect(CartesianIndices((length(rs), length(Ns))))
    progress_lock = ReentrantLock()
    completed = Ref(0)

    @threads for job in jobs
        i, j = Tuple(job)
        r = rs[i]
        Nval = Ns[j]

        params = Dict(
            "global_decay" => 1.0,
            "global_pump" => 0.0,
            "local_pump" => 0.0,
            "local_dephasing" => r * Nval,
            "local_decay" => g * Nval / log(Nval),
        )

        imc, tmc = mc_data_adaptive(Nval, params, N_traj; time_factor = time_factor)

        lock(progress_lock) do
            mc_intens[i, j] = imc
            mc_t[i, j] = tmc

            completed[] += 1
            println("Finite-N scan: r = $r [$i/$(length(rs))], N = $Nval [$j/$(length(Ns))], completed $(completed[])/$(length(jobs))")

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

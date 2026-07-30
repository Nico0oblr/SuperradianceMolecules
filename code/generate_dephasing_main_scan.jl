include("PermBasis.jl")
include("MonteCarloMF.jl")

using Base.Threads
using JLD2

function generate_dephasing_main_scan(;
    Ns = [1000, 10000, 100000, 1000000],
    N_traj = 2000,
    rs = range(0.0, 2.0, 300),
    g = 0.0,
    time_factor = 50.0,
    outfile = "../plot_data/dephasing_main_scan.jld2",
    save_progress = false,
)
    mf_intens = zeros(length(rs), length(Ns))
    mc_intens = zeros(length(rs), length(Ns))
    mf_t = zeros(length(rs), length(Ns))
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

        result = mc_vs_mf_adaptive(Nval, params, N_traj; time_factor=time_factor)

        lock(progress_lock) do
            mf_intens[i, j] = result.I_mf
            mc_intens[i, j] = result.I_mc
            mf_t[i, j] = result.t_mf
            mc_t[i, j] = result.t_mc

            completed[] += 1
            println("Main scan: r = $(round(r, digits=4)) [$i/$(length(rs))], N = $Nval [$j/$(length(Ns))], completed $(completed[])/$(length(jobs))")

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

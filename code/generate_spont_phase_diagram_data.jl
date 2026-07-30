include("PermBasis.jl")
include("MonteCarloMF.jl")

using Base.Threads
using DifferentialEquations
using JLD2

function generate_spont_phase_diagram_data(;
    outfile = "../plot_data/spont_phase_diagram.jld2",
    gs = range(0.0, 0.38, 200),
    rs = range(0.0, 1.2, 200),
    pow = 200,
    reltol = 1e-8,
    abstol = 1e-10,
)
    N = BigInt(10)^pow
    Nf = Float64(N)

    data = zeros(length(rs), length(gs))

    t_end = 100.0 * log(Nf) / Nf
    saveat = range(0.0, t_end, 30000)
    jobs = collect(CartesianIndices(data))

    @threads for job in jobs
        i, j = Tuple(job)
        r = rs[i]
        g = gs[j]
        Γ = 1.0
        ξ = Nf * r
        γ = Nf * g / log(Nf)

        p = [Γ, ξ, Nf, γ]
        u0 = [-log(Nf), 0.5]

        prob = ODEProblem(meanfield_log_transformed!, u0, (0.0, t_end), p)
        sol = solve(
            prob,
            AutoTsit5(Rosenbrock23()),
            reltol = reltol,
            abstol = abstol,
            saveat = saveat
        )

        data[i, j] = exp(maximum(x -> x[1], sol.u))
    end

    jldsave(outfile;
        gs = collect(gs),
        rs = collect(rs),
        data = data,
        pow = pow,
        N = string(N),
    )
end

function main()
    generate_spont_phase_diagram_data()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

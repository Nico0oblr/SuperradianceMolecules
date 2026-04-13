include("PermBasis.jl")
include("MonteCarloMF.jl")

using DifferentialEquations
using JLD2

function generate_spont_meanfield_g_gamma_data(;
    outfile = "../plot_data/spont_meanfield_g_gamma_data.jld2",
    powers = [5, 10, 15],
    gs = range(0.0, 0.2, 300),
    r = 0.0,
    reltol = 1e-8,
    abstol = 1e-10,
)
    curves = NamedTuple[]

    for pow in powers
        N = BigInt(10)^pow
        Nf = Float64(N)

        intens = Float64[]
        for g in gs
            Γ = 1.0
            ξ = Nf * r
            γ = Nf * g

            p = [Γ, ξ, Nf, γ]
            u0 = [-log(Nf), 0.5]

            t_end = 100.0 * log(Nf) / Nf
            prob = ODEProblem(meanfield_log_transformed!, u0, (0.0, t_end), p)
            sol = solve(
                prob,
                AutoTsit5(Rosenbrock23()),
                reltol = reltol,
                abstol = abstol,
                saveat = range(0.0, t_end, 30000)
            )

            max_log_I = maximum(x -> x[1], sol.u)
            push!(intens, exp(max_log_I))
        end

        push!(curves, (
            pow = pow,
            N = string(N),
            gs = collect(gs),
            intens = collect(intens),
        ))
    end

    data = (
        powers = collect(powers),
        gs = collect(gs),
        r = r,
        curves = curves,
    )

    jldsave(outfile; data...)
    return data
end

function main()
    generate_spont_meanfield_g_gamma_data()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
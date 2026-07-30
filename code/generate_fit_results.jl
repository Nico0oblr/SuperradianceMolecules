include("MonteCarloMF.jl")

using DifferentialEquations
using JLD2
using Statistics

"""
    powerlaw_fit_with_offset_scan(x, y; Cmin=0.0, Cmax=nothing, nC=200)

Fit `y = A*x^alpha + C` by scanning the offset `C` and performing a
log-log linear fit for each admissible offset.
"""
function powerlaw_fit_with_offset_scan(x, y; Cmin = 0.0, Cmax = nothing, nC = 200)
    Cmax === nothing && (Cmax = minimum(y) * 0.99)

    lx = log.(x)
    best = nothing
    best_err = Inf

    for C in range(Cmin, Cmax; length = nC)
        yt = y .- C
        any(yt .<= 0) && continue

        ly = log.(yt)
        alpha = cov(lx, ly) / var(lx)
        b = mean(ly) - alpha * mean(lx)
        A = exp(b)
        err = mean((ly .- (alpha .* lx .+ b)) .^ 2)

        if err < best_err
            best_err = err
            best = (A = A, alpha = alpha, C = C, err = err)
        end
    end

    return best
end

"""
    generate_fit_results(; kwargs...)

Generate the mean-field exponent scan used by `plot_beta_mc.ipynb` and save
`rs`, `gs`, `fit_data`, and `fit_data2` to `fit_results.jld2`.
"""
function generate_fit_results(;
    outfile = "../plot_data/fit_results.jld2",
    rs = range(0.0, 1.1, 100),
    gs = range(0.0, 0.5, 100),
    pows = range(23.0, 100.0, 20),
    reltol = 1e-15,
    abstol = 1e-15,
    nC = 200,
)
    rs = collect(Float64.(rs))
    gs = collect(Float64.(gs))
    pows = collect(Float64.(pows))
    Ns = 10.0 .^ pows

    fit_data = zeros(length(rs), length(gs))
    fit_data2 = zeros(length(rs), length(gs))

    for (i, r) in enumerate(rs)
        for (j, g) in enumerate(gs)
            println("Mean-field beta scan: r=$r [$i/$(length(rs))], g=$g [$j/$(length(gs))]")

            peak_intensities = Float64[]

            for pow in pows
                N = BigFloat(10)^pow
                Nf = Float64(N)
                Gamma = 1.0
                xi = Nf * r
                gamma = Nf * g / log(Nf)

                p = [Gamma, xi, Nf, gamma]
                u0 = [-log(Nf), 0.5]
                t_end = 100.0 * log(Nf) / Nf

                prob = ODEProblem(meanfield_log_transformed!, u0, (0.0, t_end), p)
                sol = solve(
                    prob,
                    AutoTsit5(Rosenbrock23());
                    reltol = reltol,
                    abstol = abstol,
                    saveat = range(0.0, t_end, 30000),
                )

                push!(peak_intensities, exp(maximum(x -> x[1], sol.u)))
            end

            fit = powerlaw_fit_with_offset_scan(Ns, peak_intensities; nC = nC)
            fit === nothing && error("No valid power-law fit for r=$r, g=$g")

            fit_data[i, j] = fit.alpha
            fit_data2[i, j] = fit.C
        end
    end

    jldsave(
        outfile;
        rs = rs,
        gs = gs,
        fit_data = fit_data,
        fit_data2 = fit_data2,
    )

    return (; rs, gs, fit_data, fit_data2)
end

function main()
    generate_fit_results()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

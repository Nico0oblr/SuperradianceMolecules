using DifferentialEquations

_plot_cache_dir = mktempdir()
ENV["MPLCONFIGDIR"] = get(ENV, "MPLCONFIGDIR", _plot_cache_dir)
ENV["XDG_CACHE_HOME"] = get(ENV, "XDG_CACHE_HOME", _plot_cache_dir)

include("PlotSetup.jl")

using PyPlot
using Printf

"""
    meanfield_bar_gamma!(du, u, p, tau)

Mean-field equations in rescaled time tau = N Gamma t, using
u = [log(Ifrak), mfrak]. The spontaneous-emission parameter is supplied as
bar_g_gamma = g_gamma log(N), so g_gamma = bar_g_gamma / log(N).
"""
function meanfield_bar_gamma!(du, u, p, tau)
    logI, m = u
    g_xi, bar_g_gamma, logN = p
    g_gamma = bar_g_gamma / logN

    du[1] = 2.0 * m - g_xi - g_gamma
    du[2] = -exp(logI) - g_gamma * (m + 0.5)

    return nothing
end

function solve_trajectory(;
    logN = 10.0 * log(10.0),
    g_xi = 0.2,
    bar_g_gamma = 0.05,
    tau_factor = 50.0,
    savepoints = 80000,
    reltol = 1e-15,
    abstol = 1e-15,
)
    u0 = [-logN, 0.5]
    tau_end = tau_factor * logN
    p = (g_xi, bar_g_gamma, logN)

    prob = ODEProblem(meanfield_bar_gamma!, u0, (0.0, tau_end), p)
    sol = solve(
        prob,
        AutoTsit5(Rosenbrock23()),
        reltol = reltol,
        abstol = abstol,
        saveat = range(0.0, tau_end, savepoints),
    )

    return sol
end

function trajectory_coordinates(sol; logN)
    tau = collect(sol.t)
    logI = [u[1] for u in sol.u]
    m = [u[2] for u in sol.u]
    Ifrak = exp.(logI)

    s = sqrt.(max.(Ifrak .+ m.^2, 0.0))
    n_t = s .+ m
    n_delta = s .- m
    depletion = 0.5 .- m
    beta = (log(4.0) .+ logI .+ 2.0 * logN) ./ logN

    peak_index = argmax(logI)

    return (
        tau = tau,
        logI = logI,
        Ifrak = Ifrak,
        s = s,
        m = m,
        n_t = n_t,
        n_delta = n_delta,
        depletion = depletion,
        beta = beta,
        peak_index = peak_index,
        peak_tau = tau[peak_index],
        peak_Ifrak = Ifrak[peak_index],
        peak_beta = beta[peak_index],
        peak_m = m[peak_index],
        peak_s = s[peak_index],
        peak_n_t = n_t[peak_index],
        peak_n_delta = n_delta[peak_index],
    )
end

"""
    parameter_sets_for_cut(; lambda_values, g_xi_start, g_xi_stop,
                            bar_g_gamma_start, bar_g_gamma_stop)

Build a one-dimensional cut through the (bar_g_gamma, g_xi) phase diagram.
The cut parameter lambda runs from 0 to 1 by default. Both parameters are
linearly interpolated between their start and stop values.
"""
function parameter_sets_for_cut(;
    lambda_values = range(0.0, 1.0, length = 11),
    g_xi_start = 0.0,
    g_xi_stop = 1.0,
    bar_g_gamma_start = 0.0,
    bar_g_gamma_stop = 1.0,
)
    return [
        (
            lambda = lambda,
            g_xi = (1.0 - lambda) * g_xi_start + lambda * g_xi_stop,
            bar_g_gamma = (1.0 - lambda) * bar_g_gamma_start + lambda * bar_g_gamma_stop,
        )
        for lambda in lambda_values
    ]
end

function add_dicke_triangle!(ax)
    ax.set_facecolor("0.90")
    ax.fill(
        [0.0, 0.5, 0.5],
        [0.0, -0.5, 0.5],
        color = "white",
        zorder = -10,
    )

    ax.plot([0.0, 0.5], [0.0, 0.5], color = "0.25", linewidth = 1.0)
    ax.plot([0.0, 0.5], [0.0, -0.5], color = "0.25", linewidth = 1.0)
    ax.plot([0.5, 0.5], [-0.5, 0.5], color = "0.25", linewidth = 1.0)

    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(-0.52, 0.52)
    ax.set_aspect(0.5, adjustable = "box")
    ax.set_xlabel(L"S/N")

    return ax
end

function box_and_connect_subthreshold_peaks!(
    ax_beta,
    ax_triangle,
    summaries;
    beta_threshold = 2.0,
    beta_threshold_tol = 1e-8,
    color = "black",
    linewidth = 1.2,
    padding_fraction = 0.18,
    minimum_width_fraction = 0.035,
    minimum_height_fraction = 0.035,
)
    selected = [
        row for row in summaries
        if row.peak_beta < beta_threshold - beta_threshold_tol
    ]

    isempty(selected) && return selected

    patches = pyimport("matplotlib.patches")

    beta_x = [row.peak_beta for row in selected]
    beta_y = [row.peak_m for row in selected]
    triangle_x = [row.peak_s for row in selected]
    triangle_y = [row.peak_m for row in selected]

    beta_box = box_around_points(
        beta_x,
        beta_y,
        ax_beta;
        padding_fraction = padding_fraction,
        minimum_width_fraction = minimum_width_fraction,
        minimum_height_fraction = minimum_height_fraction,
    )
    triangle_box = box_around_points(
        triangle_x,
        triangle_y,
        ax_triangle;
        padding_fraction = padding_fraction,
        minimum_width_fraction = minimum_width_fraction,
        minimum_height_fraction = minimum_height_fraction,
    )

    beta_patch = patches.Rectangle(
        (beta_box.xmin, beta_box.ymin),
        beta_box.width,
        beta_box.height,
        fill = false,
        edgecolor = color,
        linestyle = "--",
        linewidth = linewidth,
        zorder = 8,
    )
    triangle_patch = patches.Rectangle(
        (triangle_box.xmin, triangle_box.ymin),
        triangle_box.width,
        triangle_box.height,
        fill = false,
        edgecolor = color,
        linestyle = "--",
        linewidth = linewidth,
        zorder = 8,
    )

    ax_beta.add_patch(beta_patch)
    ax_triangle.add_patch(triangle_patch)

    for yside in (:bottom, :top)
        beta_y_anchor = yside == :bottom ? beta_box.ymin : beta_box.ymax
        triangle_y_anchor = yside == :bottom ? triangle_box.ymin : triangle_box.ymax
        connection = patches.ConnectionPatch(
            xyA = (
                beta_box.xmax,
                beta_y_anchor,
            ),
            coordsA = ax_beta.transData,
            xyB = (
                triangle_box.xmax,
                triangle_y_anchor,
            ),
            coordsB = ax_triangle.transData,
            color = color,
            linewidth = linewidth,
            linestyle = "dotted",
            zorder = 8,
        )
        ax_beta.figure.add_artist(connection)
    end

    return selected
end

function box_around_points(
    xs,
    ys,
    ax;
    padding_fraction = 0.18,
    minimum_width_fraction = 0.035,
    minimum_height_fraction = 0.035,
)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    min_width = minimum_width_fraction * abs(xlim[2] - xlim[1])
    min_height = minimum_height_fraction * abs(ylim[2] - ylim[1])

    xmin = minimum(xs)
    xmax = maximum(xs)
    ymin = minimum(ys)
    ymax = maximum(ys)
    width = max(xmax - xmin, min_width)
    height = max(ymax - ymin, min_height)
    xcenter = 0.5 * (xmin + xmax)
    ycenter = 0.5 * (ymin + ymax)
    width *= 1.0 + padding_fraction
    height *= 1.0 + padding_fraction

    return (
        xmin = xcenter - 0.5 * width,
        xmax = xcenter + 0.5 * width,
        ymin = ycenter - 0.5 * height,
        ymax = ycenter + 0.5 * height,
        width = width,
        height = height,
    )
end

function summarize_cut(;
    logN = 10.0 * log(10.0),
    parameter_sets = parameter_sets_for_cut(),
    tau_factor = 50.0,
    savepoints = 80000,
    reltol = 1e-15,
    abstol = 1e-15,
)
    summaries = NamedTuple[]

    for params in parameter_sets
        sol = solve_trajectory(;
            logN = logN,
            g_xi = params.g_xi,
            bar_g_gamma = params.bar_g_gamma,
            tau_factor = tau_factor,
            savepoints = savepoints,
            reltol = reltol,
            abstol = abstol,
        )
        coords = trajectory_coordinates(sol; logN = logN)

        push!(summaries, (
            lambda = params.lambda,
            g_xi = params.g_xi,
            bar_g_gamma = params.bar_g_gamma,
            peak_tau = coords.peak_tau,
            peak_Ifrak = coords.peak_Ifrak,
            peak_beta = coords.peak_beta,
            peak_m = coords.peak_m,
            peak_s = coords.peak_s,
            peak_n_t = coords.peak_n_t,
            peak_n_delta = coords.peak_n_delta,
        ))
    end

    return summaries
end

function write_cut_summary_csv(outfile, summaries)
    mkpath(dirname(outfile))
    open(outfile, "w") do io
        println(io, "lambda,g_xi,bar_g_gamma,peak_tau,peak_Ifrak,peak_beta,peak_m,peak_s,peak_n_t,peak_n_delta")
        for row in summaries
            @printf(
                io,
                "%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n",
                row.lambda,
                row.g_xi,
                row.bar_g_gamma,
                row.peak_tau,
                row.peak_Ifrak,
                row.peak_beta,
                row.peak_m,
                row.peak_s,
                row.peak_n_t,
                row.peak_n_delta,
            )
        end
    end

    return outfile
end

function plot_beta_and_dicke_triangle_cut(;
    outfile = joinpath(@__DIR__, "..", "data", "beta_dicke_triangle_phase_cut.pdf"),
    csv_outfile = replace(outfile, ".pdf" => ".csv"),
    logN = 10.0 * log(10.0),
    parameter_sets = parameter_sets_for_cut(),
    tau_factor = 50.0,
    savepoints = 80000,
    reltol = 1e-15,
    abstol = 1e-15,
    cmap_name = "plasma_r",
    colorbar_label = L"cut parameter $\lambda$",
    peak_beta_threshold = 2.0,
    peak_beta_threshold_tol = 1e-8,
)
    mkpath(dirname(outfile))

    lambda_values = [params.lambda for params in parameter_sets]
    norm = matplotlib.colors.Normalize(vmin = minimum(lambda_values), vmax = maximum(lambda_values))
    cmap = get_cmap(cmap_name)

    latex_plot_setup(font_size_pt = 18)
    fig = figure(figsize = (8.4, 3.5), constrained_layout = true)
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios = [1.25, 1.36, 0.07],
        wspace = 0.02,
    )
    ax_beta = fig.add_subplot(gs[1, 1])
    ax_triangle = fig.add_subplot(gs[1, 2], sharey = ax_beta)
    cax = fig.add_subplot(gs[1, 3])

    summaries = NamedTuple[]

    for params in parameter_sets
        sol = solve_trajectory(;
            logN = logN,
            g_xi = params.g_xi,
            bar_g_gamma = params.bar_g_gamma,
            tau_factor = tau_factor,
            savepoints = savepoints,
            reltol = reltol,
            abstol = abstol,
        )
        coords = trajectory_coordinates(sol; logN = logN)
        color = cmap(norm(params.lambda))

        ax_beta.plot(coords.beta, coords.m, color = color, linewidth = 1.8)
        ax_triangle.plot(coords.s, coords.m, color = color, linewidth = 1.8)

        ax_beta.scatter(
            [coords.beta[coords.peak_index]],
            [coords.m[coords.peak_index]],
            color = color,
            edgecolors = "k",
            linewidths = 0.4,
            s = 40,
            zorder = 4,
        )

        ax_triangle.scatter(
            [coords.s[coords.peak_index]],
            [coords.m[coords.peak_index]],
            color = color,
            edgecolors = "k",
            linewidths = 0.4,
            s = 40,
            zorder = 4,
        )

        push!(summaries, (
            lambda = params.lambda,
            g_xi = params.g_xi,
            bar_g_gamma = params.bar_g_gamma,
            peak_tau = coords.peak_tau,
            peak_Ifrak = coords.peak_Ifrak,
            peak_beta = coords.peak_beta,
            peak_m = coords.peak_m,
            peak_s = coords.peak_s,
            peak_n_t = coords.peak_n_t,
            peak_n_delta = coords.peak_n_delta,
        ))
    end

    ax_beta.axvline(2.0, color = "k", linestyle = "dotted", linewidth = 1.0)
    ax_beta.axvline(1.0, color = "k", linestyle = "dotted", linewidth = 1.0)
    ax_beta.set_xlim(left = 0.0, right = 2.1)
    ax_beta.set_ylim(-0.5, 0.5)
    ax_beta.set_yticks([-0.5, 0.0, 0.5])
    ax_triangle.set_yticks([-0.5, 0.0, 0.5])

    ax_beta.set_ylabel(L"M/N")
    ax_beta.set_xlabel(L"\beta(t)")

    add_dicke_triangle!(ax_triangle)
    ax_triangle.tick_params(axis = "y", left = false, labelleft = false)

    highlighted_peaks = box_and_connect_subthreshold_peaks!(
        ax_beta,
        ax_triangle,
        summaries;
        beta_threshold = peak_beta_threshold,
        beta_threshold_tol = peak_beta_threshold_tol,
    )

    sm = matplotlib.cm.ScalarMappable(norm = norm, cmap = cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax = cax)
    cbar.set_label(colorbar_label)

    fig.savefig(outfile, bbox_inches = "tight")
    close(fig)

    write_cut_summary_csv(csv_outfile, summaries)

    return (
        outfile = outfile,
        csv_outfile = csv_outfile,
        logN = logN,
        parameter_sets = parameter_sets,
        summaries = summaries,
        highlighted_peaks = highlighted_peaks,
    )
end

function main()
    Nf = 50.0

    result = plot_beta_and_dicke_triangle_cut(;
        outfile = joinpath(@__DIR__, "..", "data", "beta_dicke_triangle_phase_cut.pdf"),
        logN = Nf * log(10.0),
        parameter_sets = parameter_sets_for_cut(
            lambda_values = range(0.0, 1.0, length = 11),
            g_xi_start = 0.0,
            g_xi_stop = 1.0,
            bar_g_gamma_start = 0.0,
            bar_g_gamma_stop = 1.0,
        ),
        cmap_name = "plasma_r",
    )

    println("Saved ", result.outfile)
    println("Saved ", result.csv_outfile)
    for row in result.summaries
        @printf(
            "lambda=%.3f, g_xi=%.3f, bar_g_gamma=%.3f: peak tau=%.4g, Ifrak*=%.4g, beta*=%.4g\n",
            row.lambda,
            row.g_xi,
            row.bar_g_gamma,
            row.peak_tau,
            row.peak_Ifrak,
            row.peak_beta,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

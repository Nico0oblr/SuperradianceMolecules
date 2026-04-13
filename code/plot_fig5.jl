include("PlotSetup.jl")

using PyPlot
using JLD2

include("style.jl")

function analytic_spont_boundary(rs)
    return (1 .- rs) .- (1 .+ rs) .* log.(2 ./ (1 .+ rs))
end

function plot_spont_scaling_single!(ax, data_sharp, data_coll;
    colors_mc = ["red", "blue", "orange"],
    colors_sharp = ["blue", "green", "violet"],
    mf_linestyle = "solid",
    sharp_linestyle = "solid",
)
    gs = data_coll["gs"]
    Ns = data_coll["Ns"]
    mc = data_coll["mc_intens"]
    mf = data_coll["mf_itens"]

    for j in (1:length(Ns))[[1,3]]
        ax.scatter(
            gs[1:5:end],
            mc[:, j][1:5:end] ./ mc[1, j],
            facecolor = colors_mc[j],
            edgecolor = "k",
            s = 18,
            linewidths = 0.45,
            label = "MC " * sci_label(Ns[j])
        )

        ax.plot(
            gs,
            mf[:, j] ./ mf[1, j],
            color = colors_mc[j],
            linestyle = mf_linestyle,
            linewidth = 1.1,
            alpha = 1.0,
            label = "MF " * sci_label(Ns[j])
        )
    end

    for (i, curve) in enumerate(data_sharp["curves"][[1,2]])
        ax.plot(
            curve.gs,
            curve.intens ./ curve.intens[1],
            color = colors_sharp[i],
            linestyle = sharp_linestyle,
            linewidth = 1.8,
            alpha = 0.85,
            label = "\$N=10^{$(curve.pow)}\$ MF"
        )
    end

    ax.set_yscale("log")
    ax.set_xlabel(L"$\bar g_\gamma$")
    ax.set_ylabel(L"$I_\star(\bar g_\gamma)/I_\star(0)$")
    ax.legend(; legend_kwargs...)
    ax.set_ylim(bottom = 1e-7, top = 10)

    return ax
end

function plot_spont_phase_diagram!(
    ax,
    d;
    show_contour = true,
    contour_tol = 1e-2,
    show_analytic_boundary = true,
    rescale_intensity = 4.0,
    cmap = "viridis",
    add_colorbar = true,
    colorbar_label = L"$I/N^2$",
    boundary_color = "red",
    analytic_color = "white",
    analytic_linestyle = "--",
)
    gs = d["gs"]
    rs = d["rs"]
    data = d["data"]

    im = ax.imshow(
        rescale_intensity .* data,
        extent = [first(gs), last(gs), first(rs), last(rs)],
        origin = "lower",
        aspect = "auto",
        cmap = cmap
    )

    if show_contour
        mask = Float64.(data .<= contour_tol)
        ax.contour(
            gs, rs, mask,
            levels = [0.5],
            colors = boundary_color,
            linewidths = 2,
            zorder = 10
        )
    end

    if show_analytic_boundary
        rb = collect(rs)
        gb = analytic_spont_boundary(rb)
        valid = (gb .>= minimum(gs)) .& (gb .<= maximum(gs))
        ax.plot(
            gb[valid], rb[valid],
            color = analytic_color,
            linestyle = analytic_linestyle,
            linewidth = 2.0,
            zorder = 11
        )
    end

    ax.set_xlabel(L"$\bar g_\gamma$")
    ax.set_ylabel(L"$g_\xi$")

    if add_colorbar
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.set_title(colorbar_label)
    end

    return im
end

function plot_spont_meanfield_g_gamma!(ax, data;
    colors = ["red", "blue", "orange"],
    linestyle = "solid",
)
    for (i, curve) in enumerate(data["curves"])
        ax.plot(
            curve.gs,
            curve.intens ./ curve.intens[1],
            color = colors[i],
            linestyle = linestyle,
            linewidth = 1.8,
            alpha = 0.9,
            label = "\$N=10^{$(curve.pow)}\$"
        )
    end

    ax.set_yscale("log")
    ax.set_xlabel(L"$g_\gamma$")
    ax.set_ylabel(L"$I_\star(g_\gamma)/I_\star(0)$")
    ax.legend(; legend_kwargs...)

    return ax
end

function plot_spont_peak_times!(ax, data_coll, data_mf;
    cs = ["red", "stub", "blue", "orange", "green"],
    xlabel = L"$\bar g_{\gamma}$",
    ylabel = L"$N t_{\star} / \log N$",
    ms = 2,
    lw = 1.5,
    mc_label_prefix = "MC",
    mf_label_prefix = "MF",
    legend_kwargs = (
        fontsize = 12,
        framealpha = 0.9,
        borderpad = 0.3,
        labelspacing = 0.3,
        handlelength = 2.0,
    ),
)
    ncurves_mc = size(data_coll["mc_t"], 2)
    for j = 1:ncurves_mc
        if j == 2
            continue
        end
        N = data_coll["Ns"][j]
        ax.scatter(
            data_coll["gs"],
            data_coll["mc_t"][:, j] / log(N) * N;
            color = cs[j],
            s = ms,
        )
    end

    ncurves_mf = length(data_mf["curves"])
    for j = 1:ncurves_mf
        if j == 2
            continue
        end
        N = parse(Float64, data_mf["curves"][j].N)
        ax.plot(
            data_mf["curves"][j].gs,
            data_mf["curves"][j].max_t / log(N) * N;
            color = cs[j],
            linestyle = "solid",
            linewidth = lw,
        )
    end

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    mc_proxy = ax.scatter([], []; color = "black", s = 30)
    mf_proxy, = ax.plot([], []; color = "black", linestyle = "solid", linewidth = lw)

    leg1 = ax.legend(
        [mc_proxy, mf_proxy],
        [mc_label_prefix, mf_label_prefix];
        loc = "upper right",
        legend_kwargs...
    )

    n_proxies = Any[]
    n_labels = Any[]

    for j = 1:ncurves_mf
        if j == 2
            continue
        end
        N = parse(Float64, data_mf["curves"][j].N)

        proxy, = ax.plot(
            [],
            [];
            color = cs[j],
            linestyle = "solid",
            linewidth = lw,
        )

        push!(n_proxies, proxy)
        push!(n_labels, sci_label(N))
    end

    leg2 = ax.legend(
        n_proxies,
        n_labels;
        loc = "lower left",
        legend_kwargs...
    )

    ax.add_artist(leg1)
    ax.set_yscale(:log)

    mticker = PyPlot.matplotlib["ticker"]
    ax.set_yticks([0.6, 1.0, 2.0])
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    return ax
end

function main(;
    sharp_file = "../plot_data/spont_sharpening_data.jld2",
    collapse_file = "../plot_data/spont_collapse_data.jld2",
    ggamma_file = "../plot_data/spont_meanfield_g_gamma_data.jld2",
    phase_file = "../plot_data/spont_phase_diagram.jld2",
    times_file = "../plot_data/times_spont_meanfield.jld2",
    outfile = "../data/fig5.pdf",
)
    fig, axes = subplots(ncols = 2, nrows = 2, figsize = 2.0 .* latex_plot_setup(ratio=0.9, scale=1.8))

    data_sharp = load(sharp_file)
    data_coll = load(collapse_file)
    plot_spont_scaling_single!(axes[1,1], data_sharp, data_coll)

    data = load(ggamma_file)
    plot_spont_meanfield_g_gamma!(axes[1,2], data)
    axes[1,2].set_ylim(bottom = 1e-10, top = 10)

    d = load(phase_file)
    plot_spont_phase_diagram!(
        axes[2,1], d;
        show_contour = true,
        show_analytic_boundary = true,
        contour_tol = 1e-3,
        rescale_intensity = 4.0
    )

    data_mc = load(collapse_file)
    data_mf = load(times_file)
    plot_spont_peak_times!(axes[2,2], data_mc, data_mf)

    tight_layout()

    axes[1,1].text(-0.3, 1.0, "(a)", transform=axes[1,1].transAxes, va="top")
    axes[1,2].text(-0.25, 1.0, "(b)", transform=axes[1,2].transAxes, va="top")
    axes[2,1].text(-0.3, 1.0, "(c)", transform=axes[2,1].transAxes, va="top")
    axes[2,2].text(-0.25, 1.0, "(d)", transform=axes[2,2].transAxes, va="top")

    savefig(outfile, bbox_inches = "tight")
    println("Saved figure to: ", outfile)

    return fig, axes
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
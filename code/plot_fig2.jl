include("PlotSetup.jl")

using PyPlot
using JLD2

include("style.jl")

function plot_fig2a!(ax, data)
    for (S_bg, M_bg) in data["bg_trajectories"]
        ax.plot(S_bg, M_bg, color="k", alpha=0.85, linestyle="solid", linewidth=1.0)
    end

    ax.plot(
        data["S_mc"], data["M_mc"],
        color="blue", linestyle="solid", linewidth=2.0,
        label="MC average"
    )

    ax.plot(
        data["S_mf"], data["M_mf"],
        color="red", linestyle="dashed", linewidth=2.0,
        label="MF"
    )

    ax.plot(data["Sgrid"], data["upper_boundary"], "k--", linewidth=1.5)
    ax.plot(data["Sgrid"], data["lower_boundary"], "k--", linewidth=1.5)
    ax.plot(data["vertical_boundary_x"], data["vertical_boundary_y"], "k--", linewidth=1.5)

    ax.set_xlabel(L"$S/N$")
    ax.set_ylabel(L"$M/N$")
    ax.legend(loc="lower left"; legend_kwargs...)

    return ax
end

function plot_compare_mc_mf_intensity!(ax, data; cmap_name="viridis")
    cmap = get_cmap(cmap_name)
    rs = data["rs"]
    norm = matplotlib.colors.Normalize(vmin=minimum(rs), vmax=maximum(rs))

    for curve in data["curves"]
        r = curve.r
        color = cmap(norm(r))

        ax.plot(
            curve.ts_mc, curve.I_mc;
            color = color,
            linestyle = "-",
            linewidth = 2.0,
        )

        ax.plot(
            curve.ts_mf, curve.I_mf;
            color = color,
            linestyle = "--",
            linewidth = 1.2,
            alpha = 0.8,
        )
    end

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$t\Gamma$")
    ax.set_ylabel(L"$I/(\Gamma N^2)$")
    ax.set_ylim(1e-5, 1)

    mc_proxy, = ax.plot([], []; color = "black", linestyle = "-", linewidth = 2.0)
    mf_proxy, = ax.plot([], []; color = "black", linestyle = "--", linewidth = 1.2)

    leg1 = ax.legend(
        [mc_proxy, mf_proxy],
        [L"$\mathrm{MC}$", L"$\mathrm{MF}$"];
        loc = "lower left",
        legend_kwargs...
    )

    r_proxies = Any[]
    r_labels = Any[]

    for r in rs
        color = cmap(norm(r))
        proxy, = ax.plot([], []; color = color, linestyle = "-", linewidth = 2.0)
        push!(r_proxies, proxy)
        push!(r_labels, "\$g_\\xi=$(r)\$")
    end

    leg2 = ax.legend(
        r_proxies,
        r_labels;
        loc = "upper left",
        legend_kwargs...
    )

    ax.add_artist(leg1)

    return ax
end

function plot_dephasing_panel_a!(ax, d)
    rs = d["rs"]
    Ns = d["Ns"]
    mf_intens = d["mf_intens"]
    mc_intens = d["mc_intens"]

    ax.plot(rs, 4 .* mf_intens[:, end], color="k", label="mean field")
    for j in eachindex(Ns)
        ax.scatter(rs, mc_intens[:, j] ./ mc_intens[1, j], label=sci_label(Ns[j]))
    end

    ax.set_yscale("log")
    ax.set_ylabel(L"$I_\star/I_\star(\xi=0)$")
    ax.set_xlabel(L"$\xi/(N\Gamma)$")
    ax.grid(true)
    ax.legend(; legend_kwargs...)

    return ax
end

function plot_dephasing_panel_b!(ax, d; cs=["red", "blue", "orange"])
    rs = d["rs"]
    Ns = d["Ns"]
    mf_t = d["mf_t"]
    mc_t = d["mc_t"]

    @assert length(cs) >= length(Ns)

    for j in eachindex(Ns)
        ax.plot(rs, mf_t[:, j] .* Ns[j] ./ log(Ns[j]), color=cs[j], linestyle="solid")
        ax.scatter(rs, mc_t[:, j] .* Ns[j] ./ log(Ns[j]), label=sci_label(Ns[j]))
    end

    ax.set_xlim(left=0.4, right=1.1)
    ax.set_ylim(bottom=1.0, top=14)
    ax.set_yscale("log")
    ax.set_ylabel(L"$t_\star \Gamma N/\log(N)$")
    ax.set_xlabel(L"$\xi/(N\Gamma)$")
    ax.grid(true)

    return ax
end

function main(;
    fig2a_file = "../plot_data/fig2a_data.jld2",
    compare_file = "../plot_data/fig_compare_mc_mf_intensity.jld2",
    scan_file = "../plot_data/dephasing_main_scan.jld2",
    outfile = "../data/fig2.pdf",
)
    fig, axes = subplots(
        ncols = 2,
        nrows = 2,
        figsize = 2.0 .* latex_plot_setup(ratio=0.9, scale=1.8)
    )

    plot_fig2a!(axes[1], load(fig2a_file)["data"])
    plot_dephasing_panel_a!(axes[2], load(scan_file))
    plot_compare_mc_mf_intensity!(axes[3], load(compare_file))
    plot_dephasing_panel_b!(axes[4], load(scan_file))

    tight_layout()

    axes[1,1].text(-0.3, 1.0, "(a)", transform=axes[1,1].transAxes, va="top")
    axes[1,2].text(-0.25, 1.0, "(b)", transform=axes[1,2].transAxes, va="top")
    axes[2,1].text(-0.3, 1.0, "(c)", transform=axes[2,1].transAxes, va="top")
    axes[2,2].text(-0.25, 1.0, "(d)", transform=axes[2,2].transAxes, va="top")

    savefig(outfile, bbox_inches="tight")
    println("Saved figure to: ", outfile)

    return fig, axes
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
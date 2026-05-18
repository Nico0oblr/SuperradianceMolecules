include("PlotSetup.jl")

using PyPlot
using JLD2

include("style.jl")

function namedtuple_to_dict(nt::NamedTuple)
    Dict(string(k) => v for (k, v) in pairs(nt))
end

# --------------------------------------------------
# Panel (a)
# --------------------------------------------------

function plot_fig2b_gamma!(ax, data)
    for (S_bg, M_bg) in data["bg_trajectories"]
        ax.plot(S_bg, M_bg, color="k", alpha=0.15, linestyle="solid", linewidth=1.0)
    end

    ax.plot(data["S_mc"], data["M_mc"], color="blue", linewidth=2.0, label="MC average")
    ax.plot(data["S_mf"], data["M_mf"], color="red", linewidth=2.0, label="MF", linestyle = "dashed")

    ax.plot(data["Sgrid"], data["upper_boundary"], "k--", linewidth=1.5)
    ax.plot(data["Sgrid"], data["lower_boundary"], "k--", linewidth=1.5)
    ax.plot(data["vertical_boundary_x"], data["vertical_boundary_y"], "k--", linewidth=1.5)

    ax.set_xlabel(L"$S/N$")
    ax.set_ylabel(L"$M/N$")
    ax.legend(loc="lower left"; legend_kwargs...)

    return ax
end

# --------------------------------------------------
# Panel (b)
# --------------------------------------------------

function plot_compare_mc_mf_intensity_gamma!(ax, data; cmap_name="viridis")
    cmap = get_cmap(cmap_name)
    gbars = data["gbars_gamma"]
    norm = matplotlib.colors.Normalize(vmin=minimum(gbars), vmax=maximum(gbars))

    for curve in data["curves"]
        gbar = curve.gbar_gamma
        color = cmap(norm(gbar))

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

    mc_proxy, = ax.plot([], []; color="black", linestyle="-", linewidth=2.0)
    mf_proxy, = ax.plot([], []; color="black", linestyle="--", linewidth=1.2)

    leg1 = ax.legend(
        [mc_proxy, mf_proxy],
        [L"$\mathrm{MC}$", L"$\mathrm{MF}$"];
        loc="lower left",
        legend_kwargs...
    )

    gbar_proxies = Any[]
    gbar_labels = Any[]

    for gbar in gbars
        color = cmap(norm(gbar))
        proxy, = ax.plot([], []; color=color, linestyle="-", linewidth=2.0)
        push!(gbar_proxies, proxy)
        push!(gbar_labels, "\$\\bar g_\\gamma=$(gbar)\$")
    end

    leg2 = ax.legend(
        gbar_proxies,
        gbar_labels;
        loc="upper left",
        legend_kwargs...
    )

    ax.add_artist(leg1)

    return ax
end

# --------------------------------------------------
# Main
# --------------------------------------------------

function main(;
    traj_file = "../plot_data/fig_trajectories_gamma.jld2",
    intensity_file = "../plot_data/fig_compare_mc_mf_intensity_gamma.jld2",
    outfile = "../data/fig4.pdf"
)
    data_traj = namedtuple_to_dict(load(traj_file)["data"])
    data_int = load(intensity_file)

    w, h = latex_plot_setup(ratio=0.9, scale=1.8)
    fig, axes = subplots(ncols=2, nrows=1, figsize=(2w, h))

    plot_fig2b_gamma!(axes[1], data_traj)
    plot_compare_mc_mf_intensity_gamma!(axes[2], data_int)

    axes[2].set_ylim(bottom=1e-7)
    axes[2].set_xlim(right=1e-3)

    tight_layout()

    axes[1].text(-0.35, 1.0, "(a)", transform=axes[1].transAxes, va="top")
    axes[2].text(-0.3, 1.0, "(b)", transform=axes[2].transAxes, va="top")

    savefig(outfile, bbox_inches="tight")
    println("Saved figure to: ", outfile)

    return fig, axes
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
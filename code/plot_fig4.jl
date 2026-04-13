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
    ax.plot(data["S_mf"], data["M_mf"], color="red", linewidth=2.0, label="Mean field")

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

function plot_compare_mc_mf_intensity_gamma!(ax, data; cmap_name="viridis", show_style_note=true)
    cmap = get_cmap(cmap_name)
    gbars = data["gbars_gamma"]
    norm = matplotlib.colors.Normalize(vmin=minimum(gbars), vmax=maximum(gbars))

    for curve in data["curves"]
        gbar = curve.gbar_gamma
        color = cmap(norm(gbar))

        ax.plot(
            curve.ts_mc, curve.I_mc,
            color=color, linestyle="-", linewidth=2.0,
            label="\$\\bar g_\\gamma=$(gbar)\$"
        )

        ax.plot(
            curve.ts_mf, curve.I_mf,
            color=color, linestyle="--", linewidth=1.2, alpha=0.8
        )
    end

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$t$")
    ax.set_ylabel(L"$I/N^2$")
    ax.set_ylim(1e-5, 1)
    ax.legend(; legend_kwargs...)

    if show_style_note
        ax.text(
            0.03, 0.55,
            "solid: MC\n dashed: Mean field",
            transform=ax.transAxes,
            fontsize=12,
            bbox=Dict("facecolor"=>"white", "alpha"=>0.8)
        )
    end

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
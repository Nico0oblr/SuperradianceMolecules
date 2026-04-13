using PyPlot
using JLD2
using LsqFit

include(joinpath(@__DIR__, "style.jl"))
include(joinpath(@__DIR__, "PlotSetup.jl"))

# --------------------------------------------------
# Fit helpers
# --------------------------------------------------

function fit_A_plus_B_over_N(Ns, ys; Nmin=1e5)
    x = Float64.(Ns)
    y = Float64.(ys)

    mask = x .>= Nmin
    @assert any(mask) "No data points satisfy N >= Nmin."

    xfit = x[mask]
    yfit = y[mask]

    model(x, p) = p[1] .+ p[2] ./ x
    p0 = [yfit[end], 1.0]

    fit = curve_fit(model, xfit, yfit, p0)
    A, B = fit.param

    return (
        A = A,
        B = B,
        fit = fit,
        mask = mask,
        model = model,
        xfit = xfit,
        yfit = yfit,
    )
end

function fit_B_over_N(Ns, ys; Nmin=1e5)
    x = Float64.(Ns)
    y = Float64.(ys)

    mask = x .>= Nmin
    @assert any(mask) "No data points satisfy N >= Nmin."

    xfit = x[mask]
    yfit = y[mask]

    model(x, p) = p[1] ./ x
    p0 = [1.0]

    fit = curve_fit(model, xfit, yfit, p0)
    B = fit.param[1]

    return (
        B = B,
        fit = fit,
        mask = mask,
        model = model,
        xfit = xfit,
        yfit = yfit,
    )
end

# --------------------------------------------------
# Panel D
# --------------------------------------------------

function plot_dephasing_panel_d!(ax, d)
    rs = d["rs"]
    Ns = d["Ns"]
    mc_intens = d["mc_intens"]

    for j in eachindex(rs)
        Is = mc_intens[j, :]
        x = log.(Ns)
        y = log.(Is)
        A = [ones(length(x)) x]
        coeff = A \ y
        c, b = coeff
        a = exp(c)

        ax.scatter(
            Ns,
            Is,
            label = "\$g_\\xi=$(rs[j])\\,(\\alpha=$(round(b, digits=2)))\$"
        )
        ax.plot(Ns, a .* Ns.^b, linestyle = "-")
    end

    ax.legend(; legend_kwargs...)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(L"$I/\Gamma$")
    ax.set_xlabel(L"$N$")

    return ax
end

# --------------------------------------------------
# Peak time panel
# --------------------------------------------------

function plot_dephasing_peak_time_panel!(ax, data;
    colors_scatter = ["red", "blue"],
    colors_mf = ["orange", "cyan"],
)
    Ns = data["Ns"]
    rs = data["rs"]
    mc_t = data["mc_t"]

    @assert length(rs) >= 2
    @assert length(colors_scatter) >= 2
    @assert length(colors_mf) >= 2

    for j in 1:2
        ax.scatter(
            Ns,
            mc_t[j, :],
            color = colors_scatter[j],
            label = "MC \$g_\\xi=$(rs[j])\$"
        )

        ax.plot(
            Ns,
            log.(Ns) ./ Ns ./ (1.0 .- rs[j]),
            color = colors_mf[j],
            linestyle = "solid",
            label = "MF \$g_\\xi=$(rs[j])\$"
        )
    end

    ax.legend(; legend_kwargs...)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$N$")
    ax.set_ylabel(L"$t_\star$")

    return ax
end

# --------------------------------------------------
# Below-threshold fit panel
# --------------------------------------------------

function plot_dephasing_panel_c_below_fit!(ax, d; which=2, Nmin=1e5)
    rs = d["rs"]
    Ns = d["Ns"]
    mc_intens = d["mc_intens"]

    below_inds = findall(<(1), rs)
    @assert length(below_inds) >= which "Need at least $which values with g_xi < 1."
    j = below_inds[which]

    r = rs[j]
    x = Float64.(Ns)
    y = Float64.(mc_intens[j, :] ./ Ns.^2)

    fitdata = fit_A_plus_B_over_N(x, y; Nmin=Nmin)
    A, B = fitdata.A, fitdata.B
    mask = fitdata.mask
    model = fitdata.model

    ax.scatter(
        x, y,
        label = "\$g_\\xi=$(r)\$"
    )

    ax.scatter(
        x[mask], y[mask],
        facecolor = "red",
        edgecolors = "k",
        linewidths = 0.8,
        zorder = 3
    )

    xline = collect(range(minimum(x[mask]), maximum(x), length=400))
    yline = model(xline, [A, B])

    ax.plot(
        xline, yline,
        color = "k",
        linestyle = "-",
        linewidth = 1.8,
        label = "\$A+B/N\$ fit",
        zorder = 10
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$N$")
    ax.set_ylabel(L"$I_{\max}/N^2$")
    ax.legend(; legend_kwargs...)

    return (
        ax = ax,
        r = r,
        A = A,
        B = B,
        fit = fitdata.fit,
        mask = mask,
    )
end

# --------------------------------------------------
# Above-threshold fit panel
# --------------------------------------------------

function plot_dephasing_panel_c_above_fit!(ax, d; which=2, Nmin=1e5)
    rs = d["rs"]
    Ns = d["Ns"]
    mc_intens = d["mc_intens"]

    above_inds = findall(>(1), rs)
    @assert length(above_inds) >= which "Need at least $which values with g_xi > 1."
    j = above_inds[which]

    r = rs[j]
    x = Float64.(Ns)
    y = Float64.(mc_intens[j, :] ./ Ns.^2)

    fitdata = fit_B_over_N(x, y; Nmin=Nmin)
    B = fitdata.B
    mask = fitdata.mask
    model = fitdata.model

    ax.scatter(
        x, y,
        label = "\$g_\\xi=$(r)\$"
    )

    ax.scatter(
        x[mask], y[mask],
        facecolor = "red",
        edgecolors = "k",
        linewidths = 0.8,
        zorder = 3
    )

    xline = collect(range(minimum(x[mask]), maximum(x), length=400))
    yline = model(xline, [B])

    ax.plot(
        xline, yline,
        color = "k",
        linestyle = "-",
        linewidth = 1.8,
        label = "\$B/N\$ fit",
        zorder = 10
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$N$")
    ax.set_ylabel(L"$I_{\max}/N^2$")
    ax.legend(; legend_kwargs...)

    return (
        ax = ax,
        r = r,
        B = B,
        fit = fitdata.fit,
        mask = mask,
    )
end

# --------------------------------------------------
# Main
# --------------------------------------------------

function main(; datafile=nothing, outfile=nothing)
    if datafile === nothing
        datafile = joinpath(@__DIR__, "..", "plot_data", "dephasing_finiteN_scan.jld2")
    end

    if outfile === nothing
        outfile = joinpath(@__DIR__, "..", "data", "fig3.pdf")
    end

    d = load(datafile)

    fig, axes = subplots(
        ncols = 2,
        nrows = 2,
        figsize = 2.0 .* latex_plot_setup(ratio=0.9, scale=1.8)
    )

    plot_dephasing_panel_d!(axes[1,1], d)
    plot_dephasing_peak_time_panel!(axes[1,2], d)
    plot_dephasing_panel_c_below_fit!(axes[2,1], d; which = 2)
    plot_dephasing_panel_c_above_fit!(axes[2,2], d; which = 1)

    tight_layout()

    axes[1,1].text(-0.35, 1.0, "(a)", transform=axes[1,1].transAxes, va="top")
    axes[1,2].text(-0.3, 1.0, "(b)", transform=axes[1,2].transAxes, va="top")
    axes[2,1].text(-0.35, 1.0, "(c)", transform=axes[2,1].transAxes, va="top")
    axes[2,2].text(-0.3, 1.0, "(d)", transform=axes[2,2].transAxes, va="top")

    savefig(outfile, bbox_inches = "tight")
    println("Saved figure to: ", outfile)

    return fig, axes
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
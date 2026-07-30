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

function nearest_r_index(rs, target)
    distances = abs.(Float64.(rs) .- Float64(target))
    return argmin(distances)
end

function selected_r_indices(rs, targets, predicate)
    inds = Int[]
    for target in targets
        j = nearest_r_index(rs, target)
        @assert predicate(rs[j]) "Nearest available g_xi=$(rs[j]) does not match requested side for target $target."
        push!(inds, j)
    end

    return unique(inds)
end

function color_positions(n)
    n == 1 && return [0.5]
    return collect(range(0.15, 0.85, length=n))
end

# --------------------------------------------------
# Panel D
# --------------------------------------------------

function plot_dephasing_panel_d!(ax, d)
    rs = d["rs"]
    Ns = d["Ns"]
    mc_intens = d["mc_intens"]

    for j in eachindex(rs)[[1,3,4,6]]
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
            label = "\$g_\\xi=$(rs[j])\\,(\\beta=$(round(b, digits=2)))\$"
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
    colors_scatter = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
 "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"],
    colors_mf = colors_scatter,
)
    Ns = data["Ns"]
    rs = data["rs"]
    mc_t = data["mc_t"]

    @assert length(rs) >= 3
    @assert size(mc_t, 1) >= 3
    @assert length(colors_scatter) >= 3
    @assert length(colors_mf) >= 3

    for j in 1:3
        ax.scatter(
            Ns,
            mc_t[j, :];
            color = colors_scatter[j],
        )

        ax.plot(
            Ns,
            log.(Ns) ./ Ns ./ (1.0 .- rs[j]);
            color = colors_mf[j],
            linestyle = "-",
            linewidth = 1.5,
        )
    end

    mc_proxy = ax.scatter(
        Float64[],
        Float64[];
        color = "black",
    )

    mf_proxy, = ax.plot(
        Float64[],
        Float64[];
        color = "black",
        linestyle = "-",
        linewidth = 1.5,
    )

    leg1 = ax.legend(
        [mc_proxy, mf_proxy],
        [L"$\mathrm{MC}$", L"$\mathrm{MF}$"];
        loc = "lower left",
        legend_kwargs...
    )

    r_proxies = Any[]
    r_labels = String[]

    for j in 1:3
        proxy, = ax.plot(
            Float64[],
            Float64[];
            color = colors_scatter[j],
            linestyle = "-",
            linewidth = 2.0,
        )

        push!(r_proxies, proxy)
        push!(r_labels, "\$g_\\xi=$(rs[j])\$")
    end

    ax.legend(
        r_proxies,
        r_labels;
        loc = "upper left",
        legend_kwargs...
    )

    ax.add_artist(leg1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$N$")
    ax.set_ylabel(L"$t_\star\Gamma$")

    return ax
end

# --------------------------------------------------
# Below-threshold fit panel
# --------------------------------------------------

function plot_dephasing_panel_c_below_fit!(ax, d;
    which = nothing,
    rs_fit = [0.6, 0.8, 0.95],
    Nmin = 1e5,
)
    rs = d["rs"]
    Ns = d["Ns"]
    mc_intens = d["mc_intens"]

    x = Float64.(Ns)
    below_inds = rs_fit === nothing ? [findall(<(1), rs)[which]] : selected_r_indices(rs, rs_fit, <(1))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    results = NamedTuple[]

    for (k, j) in enumerate(below_inds)
        r = rs[j]
        y = Float64.(mc_intens[j, :] ./ Ns.^2)

        fitdata = fit_A_plus_B_over_N(x, y; Nmin=Nmin)
        A, B = fitdata.A, fitdata.B
        mask = fitdata.mask
        model = fitdata.model

        color = colors[k]
        ax.scatter(
            x, y,
            color = color,
            label = "\$g_\\xi=$(r)\$"
        )

        ax.scatter(
            x[mask], y[mask],
            facecolor = color,
            edgecolors = "k",
            linewidths = 0.8,
            zorder = 3
        )

        xline = collect(range(minimum(x[mask]), maximum(x), length=400))
        yline = model(xline, [A, B])

        ax.plot(
            xline, yline,
            color = color,
            linestyle = "-",
            linewidth = 1.8,
            label = k == length(below_inds) ? "\$A+B/N\$ fits" : nothing,
            zorder = 10
        )

        push!(results, (
            r = r,
            A = A,
            B = B,
            fit = fitdata.fit,
            mask = mask,
        ))
    end

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$N$")
    ax.set_ylabel(L"$I_{\star}/(\Gamma N^2)$")
    ax.legend(; legend_kwargs...)

    return (
        ax = ax,
        fits = results,
    )
end

# --------------------------------------------------
# Above-threshold fit panel
# --------------------------------------------------

function plot_dephasing_panel_c_above_fit!(ax, d;
    which = nothing,
    rs_fit = [1.05, 1.1, 1.15],
    Nmin = 1e5,
)
    rs = d["rs"]
    Ns = d["Ns"]
    mc_intens = d["mc_intens"]

    x = Float64.(Ns)
    above_inds = rs_fit === nothing ? [findall(>(1), rs)[which]] : selected_r_indices(rs, rs_fit, >(1))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    results = NamedTuple[]

    for (k, j) in enumerate(above_inds)
        r = rs[j]
        y = Float64.(mc_intens[j, :] ./ Ns.^2)

        fitdata = fit_B_over_N(x, y; Nmin=Nmin)
        B = fitdata.B
        mask = fitdata.mask
        model = fitdata.model

        color = colors[k]
        ax.scatter(
            x, y,
            color = color,
            label = "\$g_\\xi=$(r)\$"
        )

        ax.scatter(
            x[mask], y[mask],
            facecolor = color,
            edgecolors = "k",
            linewidths = 0.8,
            zorder = 3
        )

        xline = collect(range(minimum(x[mask]), maximum(x), length=400))
        yline = model(xline, [B])

        ax.plot(
            xline, yline,
            color = color,
            linestyle = "-",
            linewidth = 1.8,
            label = k == length(above_inds) ? "\$B/N\$ fits" : nothing,
            zorder = 10
        )

        push!(results, (
            r = r,
            B = B,
            fit = fitdata.fit,
            mask = mask,
        ))
    end

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"$N$")
    ax.set_ylabel(L"$I_{\star}/(\Gamma N^2)$")
    ax.legend(; legend_kwargs...)

    return (
        ax = ax,
        fits = results,
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
    plot_dephasing_panel_c_below_fit!(axes[2,1], d)
    plot_dephasing_panel_c_above_fit!(axes[2,2], d)

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

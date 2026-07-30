using PyPlot
using JLD2
using LsqFit

include(joinpath(@__DIR__, "PlotSetup.jl"))
include(joinpath(@__DIR__, "style.jl"))

# -----------------------------------------------------------------------------
# Appendix dephasing panels
# -----------------------------------------------------------------------------

function plot_fig2a!(ax, data; cmap_name="viridis")
    cmap = get_cmap(cmap_name)
    g_xis = data["g_xis"]
    norm = matplotlib.colors.Normalize(vmin=minimum(g_xis), vmax=maximum(g_xis))

    for curve in data["curves"]
        color = cmap(norm(curve.g_xi))

        ax.plot(
            curve.S_mc, curve.M_mc;
            color = color,
            linestyle = "-",
            linewidth = 2.0,
        )

        ax.plot(
            curve.S_mf, curve.M_mf;
            color = color,
            linestyle = "--",
            linewidth = 1.2,
            alpha = 0.8,
        )
    end

    ax.plot(data["Sgrid"], data["upper_boundary"], "k--", linewidth=1.5)
    ax.plot(data["Sgrid"], data["lower_boundary"], "k--", linewidth=1.5)
    ax.plot(data["vertical_boundary_x"], data["vertical_boundary_y"], "k--", linewidth=1.5)

    ax.set_xlabel(L"$S/N$")
    ax.set_ylabel(L"$M/N$")

    mc_proxy, = ax.plot([], []; color = "black", linestyle = "-", linewidth = 2.0)
    mf_proxy, = ax.plot([], []; color = "black", linestyle = "--", linewidth = 1.2)

    leg1 = ax.legend(
        [mc_proxy, mf_proxy],
        [L"$\mathrm{MC}$", L"$\mathrm{MF}$"];
        loc = "lower left",
        legend_kwargs...
    )

    g_proxies = Any[]
    g_labels = Any[]

    for g_xi in g_xis
        color = cmap(norm(g_xi))
        proxy, = ax.plot([], []; color = color, linestyle = "-", linewidth = 2.0)
        push!(g_proxies, proxy)
        push!(g_labels, "\$g_\\xi=$(g_xi)\$")
    end

    leg2 = ax.legend(
        g_proxies,
        g_labels;
        loc = "upper left",
        legend_kwargs...
    )

    ax.add_artist(leg1)

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

    ax.plot(rs, 4 .* mf_intens[:, end], color="k", label="MF \$N=10^6\$")
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

function plot_dephasing_panel_b!(ax, d)
    rs = d["rs"]
    Ns = d["Ns"]
    mf_t = d["mf_t"]
    mc_t = d["mc_t"]

    for j in eachindex(Ns)
        ax.scatter(rs, mc_t[:, j] .* Ns[j] ./ log(Ns[j]), label=sci_label(Ns[j]))
    end

    j = length(Ns)
    ax.plot(rs, mf_t[:, j] .* Ns[j] ./ log(Ns[j]), linestyle="solid", label="mean-field", color = "k")

    ax.set_xlim(left=0.4, right=1.1)
    ax.set_ylim(bottom=1.0, top=14)
    ax.set_yscale("log")
    ax.set_ylabel(L"$t_\star \Gamma N/\log(N)$")
    ax.set_xlabel(L"$\xi/(N\Gamma)$")
    ax.grid(true)

    return ax
end

# -----------------------------------------------------------------------------
# Finite-size dephasing panels
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Local-decay panels
# -----------------------------------------------------------------------------

function gamma_color(gbar, gbars, i; cmap_name="plasma")
    if cmap_name == "plasma"
        cmap = get_cmap(cmap_name)
        norm = matplotlib.colors.Normalize(vmin=minimum(gbars), vmax=maximum(gbars))
        return cmap(norm(gbar))
    end

    return tab_colors[mod1(i, length(tab_colors))]
end

# --------------------------------------------------
# Panel (a)
# --------------------------------------------------

function plot_fig2b_gamma!(ax, data; cmap_name="plasma")
    gbars = data["gbars_gamma"]

    for (i, curve) in enumerate(data["curves"])
        color = gamma_color(curve.gbar_gamma, gbars, i; cmap_name)

        ax.plot(
            curve.S_mc, curve.M_mc;
            color = color,
            linestyle = "-",
            linewidth = 2.0,
        )

        ax.plot(
            curve.S_mf, curve.M_mf;
            color = color,
            linestyle = "--",
            linewidth = 1.2,
            alpha = 0.8,
        )
    end

    ax.plot(data["Sgrid"], data["upper_boundary"], "k--", linewidth=1.5)
    ax.plot(data["Sgrid"], data["lower_boundary"], "k--", linewidth=1.5)
    ax.plot(data["vertical_boundary_x"], data["vertical_boundary_y"], "k--", linewidth=1.5)

    ax.set_xlabel(L"$S/N$")
    ax.set_ylabel(L"$M/N$")

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

    for (i, gbar) in enumerate(gbars)
        color = gamma_color(gbar, gbars, i; cmap_name)
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
# Panel (b)
# --------------------------------------------------

function plot_compare_mc_mf_intensity_gamma!(ax, data; cmap_name="plasma")
    gbars = data["gbars_gamma"]

    for (i, curve) in enumerate(data["curves"])
        gbar = curve.gbar_gamma
        color = gamma_color(gbar, gbars, i; cmap_name)

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

    for (i, gbar) in enumerate(gbars)
        color = gamma_color(gbar, gbars, i; cmap_name)
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

# -----------------------------------------------------------------------------
# Spontaneous-emission panels
# -----------------------------------------------------------------------------

function plot_spont_scaling_single!(ax, data_sharp, data_coll;
    colors_mc = tab_colors,
    colors_sharp = tab_colors,
    mf_linestyle = "solid",
    sharp_linestyle = "solid",
)
    gs = data_coll["gs"]
    Ns = data_coll["Ns"]
    mc = data_coll["mc_intens"]
    mf = data_coll["mf_itens"]

    n_proxies = Any[]
    n_labels = Any[]

    for j in (1:length(Ns))[[1,3]]
        ax.scatter(
            gs[1:5:end],
            mc[:, j][1:5:end] ./ mc[1, j],
            facecolor = colors_mc[j],
            edgecolor = "k",
            s = 18,
            linewidths = 0.45,
        )

        ax.plot(
            gs,
            mf[:, j] ./ mf[1, j],
            color = colors_mc[j],
            linestyle = mf_linestyle,
            linewidth = 1.1,
            alpha = 1.0,
        )

        proxy, = ax.plot([], []; color = colors_mc[j], linestyle = "solid", linewidth = 1.8)
        push!(n_proxies, proxy)
        push!(n_labels, sci_label(Ns[j]))
    end

    for (i, curve) in enumerate(data_sharp["curves"][[1,2]])
        ax.plot(
            curve.gs,
            curve.intens ./ curve.intens[1],
            color = colors_sharp[i],
            linestyle = sharp_linestyle,
            linewidth = 1.8,
            alpha = 0.85,
        )

        proxy, = ax.plot([], []; color = colors_sharp[i], linestyle = "solid", linewidth = 1.8)
        push!(n_proxies, proxy)
        push!(n_labels, "\$N=10^{$(curve.pow)}\$")
    end

    ax.set_yscale("log")
    ax.set_xlabel(L"$\gamma \log N/(N\Gamma)$")
    ax.set_ylabel(L"$I_\star/I_\star(\gamma=0)$")

    mc_proxy = ax.scatter([], []; facecolor = "black", edgecolor = "k", s = 18, linewidths = 0.45)
    mf_proxy, = ax.plot([], []; color = "black", linestyle = mf_linestyle, linewidth = 1.2)

    leg1 = ax.legend(
        [mc_proxy, mf_proxy],
        [L"$\mathrm{MC}$", L"$\mathrm{MF}$"];
        loc = "lower left",
        legend_kwargs...
    )

    leg2 = ax.legend(
        n_proxies,
        n_labels;
        loc = "lower right",
        legend_kwargs...
    )

    ax.add_artist(leg1)
    ax.set_ylim(bottom = 1e-7, top = 10)

    return ax
end

function plot_spont_peak_times!(ax, data_coll, data_mf;
    cs = tab_colors,
    shown_powers = [4, 5, 6, 10, 23],
    xlabel = L"$\gamma \log N/(N\Gamma)$",
    ylabel = L"$N \Gamma t_{\star} /\log N$",
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
    mc_by_N = Dict(Float64(data_coll["Ns"][j]) => j for j in 1:ncurves_mc)

    shown_mf_curves = [
        (N = parse(Float64, curve.N), curve = curve)
        for curve in data_mf["curves"]
        if any(pow -> isapprox(Float64(curve.pow), Float64(pow); atol = 1e-10), shown_powers)
    ]

    for (i_color, item) in enumerate(shown_mf_curves)
        N = item.N
        if haskey(mc_by_N, N)
            j = mc_by_N[N]
            ax.scatter(
                data_coll["gs"],
                data_coll["mc_t"][:, j] / log(N) * N;
                color = cs[mod1(i_color, length(cs))],
                s = ms,
            )
        end
    end

    for (i_color, item) in enumerate(shown_mf_curves)
        N = item.N
        curve = item.curve
        ax.plot(
            curve.gs,
            curve.max_t / log(N) * N;
            color = cs[mod1(i_color, length(cs))],
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

    for (i_color, item) in enumerate(shown_mf_curves)
        proxy, = ax.plot(
            [],
            [];
            color = cs[mod1(i_color, length(cs))],
            linestyle = "solid",
            linewidth = lw,
        )

        push!(n_proxies, proxy)
        push!(n_labels, sci_label(item.N))
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

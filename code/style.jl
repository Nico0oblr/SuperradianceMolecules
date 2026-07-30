# -----------------------------
# Global plotting style
# -----------------------------

const legend_kwargs = (
    fontsize = 12,
    framealpha = 0.9,
    borderpad = 0.3,
    labelspacing = 0.3,
    handlelength = 2.0,
)

const tab_colors = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]

function sci_label(N)
    exp = floor(Int, log10(N))
    mant = N / 10.0^exp

    # if mantissa is (numerically) an integer → print as Int
    if isapprox(mant, round(mant); atol=1e-4)
        mant_str = string(Int(round(mant)))
        if Int(round(mant)) == 1
            return "\$N=10^{$exp}\$"
        end
    else
        mant_str = string(mant)
    end

    return "\$N=$(mant_str)\\times 10^{$exp}\$"
end

function namedtuple_to_dict(nt::NamedTuple)
    Dict(string(k) => v for (k, v) in pairs(nt))
end

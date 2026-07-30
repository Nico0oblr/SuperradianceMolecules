function analytic_spont_boundary(rs)
    return (1 .- rs) .- (1 .+ rs) .* log.(2 ./ (1 .+ rs))
end

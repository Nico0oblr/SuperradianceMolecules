include("generate_spont_sharpening_data.jl")

function main()
    generate_spont_sharpening_data(
        outfile = "../plot_data/times_spont_meanfield.jld2",
        powers = [4, 5, 6, 10, 23]
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

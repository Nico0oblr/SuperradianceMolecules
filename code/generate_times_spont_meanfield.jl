include("generate_spont_sharpening_data.jl")

function main()
    generate_spont_sharpening_data(
        outfile = "../plot_data/times_spont_meanfield.jld2",
        powers = [3, 4, 4.69897000434, 10, 23]
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
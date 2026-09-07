JULIA := julia --project=..

CODE := .
PLOT_DATA := ../plot_data

GEN_COMMON := $(CODE)/MonteCarloMF.jl

.PHONY: all data

all: data

data: \
	$(PLOT_DATA)/beta_mc_scan_data.jld2 \
	$(PLOT_DATA)/fit_results.jld2 \
	$(PLOT_DATA)/fig2a_data.jld2 \
	$(PLOT_DATA)/fig_compare_mc_mf_intensity.jld2 \
	$(PLOT_DATA)/dephasing_main_scan.jld2 \
	$(PLOT_DATA)/dephasing_finiteN_scan.jld2 \
	$(PLOT_DATA)/fig_trajectories_gamma.jld2 \
	$(PLOT_DATA)/fig_compare_mc_mf_intensity_gamma.jld2 \
	$(PLOT_DATA)/spont_sharpening_data.jld2 \
	$(PLOT_DATA)/spont_collapse_data.jld2 \
	$(PLOT_DATA)/times_spont_meanfield.jld2 \
	$(PLOT_DATA)/spont_phase_diagram_8.jld2

# -----------------------------
# Directory creation
# -----------------------------

$(PLOT_DATA):
	mkdir -p $(PLOT_DATA)

$(PLOT_DATA)/beta_mc_scan_data.jld2: $(CODE)/generate_beta_mc_scan_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/fit_results.jld2: $(CODE)/generate_fit_results.jl $(CODE)/MonteCarloMF.jl | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/fig2a_data.jld2: $(CODE)/generate_fig2a_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/fig_compare_mc_mf_intensity.jld2: $(CODE)/generate_compare_mc_mf_intensity_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/dephasing_main_scan.jld2: $(CODE)/generate_dephasing_main_scan.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/dephasing_finiteN_scan.jld2: $(CODE)/generate_dephasing_finiteN_scan.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/fig_trajectories_gamma.jld2: $(CODE)/generate_fig2b_gamma_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/fig_compare_mc_mf_intensity_gamma.jld2: $(CODE)/generate_compare_mc_mf_intensity_gamma_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/spont_sharpening_data.jld2: $(CODE)/generate_spont_sharpening_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/spont_collapse_data.jld2: $(CODE)/generate_spont_collapse_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/times_spont_meanfield.jld2: $(CODE)/generate_times_spont_meanfield.jl $(CODE)/generate_spont_sharpening_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/spont_phase_diagram_8.jld2: $(CODE)/generate_spont_phase_diagram_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

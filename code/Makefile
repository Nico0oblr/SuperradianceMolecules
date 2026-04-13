JULIA := julia

CODE := .
PLOT_DATA := ../plot_data
FIG_DATA := ../data

# Shared source dependencies
GEN_COMMON := $(CODE)/PermBasis.jl $(CODE)/MonteCarloMF.jl
PLOT_COMMON := $(CODE)/PlotSetup.jl $(CODE)/style.jl

.PHONY: all data figs clean fig2 fig3 fig4 fig5

all: figs

data: \
	$(PLOT_DATA)/fig2a_data.jld2 \
	$(PLOT_DATA)/fig_compare_mc_mf_intensity.jld2 \
	$(PLOT_DATA)/dephasing_main_scan.jld2 \
	$(PLOT_DATA)/dephasing_finiteN_scan.jld2 \
	$(PLOT_DATA)/fig_trajectories_gamma.jld2 \
	$(PLOT_DATA)/fig_compare_mc_mf_intensity_gamma.jld2 \
	$(PLOT_DATA)/spont_sharpening_data.jld2 \
	$(PLOT_DATA)/spont_collapse_data.jld2 \
	$(PLOT_DATA)/spont_phase_diagram.jld2 \
	$(PLOT_DATA)/spont_meanfield_g_gamma_data.jld2 \
	$(PLOT_DATA)/times_spont_meanfield.jld2

figs: \
	$(FIG_DATA)/fig2.pdf \
	$(FIG_DATA)/fig3.pdf \
	$(FIG_DATA)/fig4.pdf \
	$(FIG_DATA)/fig5.pdf

fig2: $(FIG_DATA)/fig2.pdf
fig3: $(FIG_DATA)/fig3.pdf
fig4: $(FIG_DATA)/fig4.pdf
fig5: $(FIG_DATA)/fig5.pdf

# -----------------------------
# Directory creation
# -----------------------------

$(PLOT_DATA):
	mkdir -p $(PLOT_DATA)

$(FIG_DATA):
	mkdir -p $(FIG_DATA)

# -----------------------------
# Figure 2 data
# -----------------------------

$(PLOT_DATA)/fig2a_data.jld2: $(CODE)/generate_fig2a_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/fig_compare_mc_mf_intensity.jld2: $(CODE)/generate_compare_mc_mf_intensity_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/dephasing_main_scan.jld2: $(CODE)/generate_dephasing_main_scan.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

# -----------------------------
# Figure 3 data
# -----------------------------

$(PLOT_DATA)/dephasing_finiteN_scan.jld2: $(CODE)/generate_dephasing_finiteN_scan.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

# -----------------------------
# Figure 4 data
# -----------------------------

$(PLOT_DATA)/fig_trajectories_gamma.jld2: $(CODE)/generate_fig2b_gamma_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/fig_compare_mc_mf_intensity_gamma.jld2: $(CODE)/generate_compare_mc_mf_intensity_gamma_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

# -----------------------------
# Figure 5 data
# -----------------------------

$(PLOT_DATA)/spont_sharpening_data.jld2: $(CODE)/generate_spont_sharpening_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/spont_collapse_data.jld2: $(CODE)/generate_spont_collapse_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/spont_phase_diagram.jld2: $(CODE)/generate_spont_phase_diagram_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/spont_meanfield_g_gamma_data.jld2: $(CODE)/generate_spont_meanfield_g_gamma_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

$(PLOT_DATA)/times_spont_meanfield.jld2: $(CODE)/generate_times_spont_meanfield.jl $(CODE)/generate_spont_sharpening_data.jl $(GEN_COMMON) | $(PLOT_DATA)
	$(JULIA) $<

# -----------------------------
# Plots
# -----------------------------

$(FIG_DATA)/fig2.pdf: $(CODE)/plot_fig2.jl $(PLOT_COMMON) \
	$(PLOT_DATA)/fig2a_data.jld2 \
	$(PLOT_DATA)/fig_compare_mc_mf_intensity.jld2 \
	$(PLOT_DATA)/dephasing_main_scan.jld2 | $(FIG_DATA)
	$(JULIA) $<

$(FIG_DATA)/fig3.pdf: $(CODE)/plot_fig3.jl $(PLOT_COMMON) \
	$(PLOT_DATA)/dephasing_finiteN_scan.jld2 | $(FIG_DATA)
	$(JULIA) $<

$(FIG_DATA)/fig4.pdf: $(CODE)/plot_fig4.jl $(PLOT_COMMON) \
	$(PLOT_DATA)/fig_trajectories_gamma.jld2 \
	$(PLOT_DATA)/fig_compare_mc_mf_intensity_gamma.jld2 | $(FIG_DATA)
	$(JULIA) $<

$(FIG_DATA)/fig5.pdf: $(CODE)/plot_fig5.jl $(PLOT_COMMON) \
	$(PLOT_DATA)/spont_sharpening_data.jld2 \
	$(PLOT_DATA)/spont_collapse_data.jld2 \
	$(PLOT_DATA)/spont_phase_diagram.jld2 \
	$(PLOT_DATA)/spont_meanfield_g_gamma_data.jld2 \
	$(PLOT_DATA)/times_spont_meanfield.jld2 | $(FIG_DATA)
	$(JULIA) $<

# -----------------------------
# Cleanup
# -----------------------------

clean:
	rm -f $(PLOT_DATA)/*.jld2 $(FIG_DATA)/*.pdf
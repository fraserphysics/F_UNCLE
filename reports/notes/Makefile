# notes.pdf is the default target
SHELL=/bin/bash
notes.pdf:
FUNCLE_ROOT = ../..

# Block to copy drawings we don't know how to make
STAT_DOC = $(FUNCLE_ROOT)/docs/_static

figs/stick_notes.pdf:
	mkdir -p figs
	cp $(STAT_DOC)/stick.pdf $@

figs/gun_notes.pdf:
	mkdir -p figs
	cp $(STAT_DOC)/gun.pdf $@

# Block to make figures from simulations
PYTHON = source activate funcle3; export PYTHONPATH=${FUNCLE_ROOT}/F_UNCLE/; \
python

SIM_FIGS = eos_diff eos eos_basis info_gun info_stick info_sphere gun_sens \
stick_sens sphere_sens conv stick_results gun_results sphere_results rayl_line

sim_figs_made: ${FUNCLE_ROOT}/examples/make_notes.py
	mkdir -p figs
	${PYTHON} $< --fig_dir $(PWD)/figs/  $(patsubst %, --%, $(SIM_FIGS))
	touch $@

SPECIAL =  stick_notes gun_notes
FIGURES_MADE = $(patsubst %, figs/%.pdf, $(SPECIAL)) sim_figs_made

notes.bbl: notes.tex ${FIGURES_MADE} local.bib
	pdflatex notes
	bibtex notes
notes.pdf: notes.bbl
	pdflatex notes
	pdflatex notes

clean:
	rm -rf figs *.pdf *.aux *.bbl *.log *.blg sim_figs_made
###---------------
### Local Variables:
### mode: makefile
### End:

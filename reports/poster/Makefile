# conda create -n something FixMe
# source activate something FixMe
PYTHON = export PYTHONPATH=../../F_UNCLE/;  python
# poster.pdf is the default target
poster.pdf:

# Block to fetch old Man From U.N.C.L.E picture
ILLYA=Robert_Vaughn_David_McCallum_Man_from_UNCLE_1966.JPG
figs/authors.jpg:
	mkdir -p figs
	wget https://upload.wikimedia.org/wikipedia/commons/8/8b/${ILLYA}
	mv ${ILLYA} $@

# Block to make figures from simulations
SIM_FIGS = stick_xt stick_fisher stick_CJ eos_nom_true gun_tv gun_fisher basis
sim_figs: plot.py
	mkdir -p figs
	${PYTHON} plot.py $(patsubst %, --% figs, $(SIM_FIGS))
	touch $@
figs/%.pdf: plot.py
	mkdir -p figs
	${PYTHON} plot.py --$* figs
FIGURES_MADE = figs/authors.jpg sim_figs

poster.pdf: poster.tex ${FIGURES_MADE}
	pdflatex poster

clean:
	rm -f *.aux *.ps *.log figs/*.pdf *.pdf sim_figs
###---------------
### Local Variables:
### mode: makefile
### End:

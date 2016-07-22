# poster.pdf is the default target
poster.pdf:

# Block to fetch old Man From U.N.C.L.E picture
ILLYA=Robert_Vaughn_David_McCallum_Man_from_UNCLE_1966.JPG
figs/authors.jpg:
	wget https://upload.wikimedia.org/wikipedia/commons/8/8b/${ILLYA}
	mv ${ILLYA} $@

# Block to copy drawings we don't know how to make
OLD_DIR = /home/andy/tmp/scipy_proceedings/poster
figs/stick_poster.pdf: ${OLD_DIR}/stick_poster.pdf
	cp $< $@
figs/gun_poster.pdf: ${OLD_DIR}/gun_poster.pdf
	cp $< $@

# Copy old fig.  FixMe: Should make that here
figs/basis.pdf: /home/andy/projects/metfie/like_gun/basis.pdf
	ln -s $< $@

# Block to make figures from simulations
PYTHON = export PYTHONPATH=/home/andy/projects/F_UNCLE/F_UNCLE/; python
SIM_FIGS = stick_xt stick_fisher stick_CJ eos_nom_true gun_tv gun_fisher
sim_figs: plot.py
	${PYTHON} plot.py $(patsubst %, --% figs, $(SIM_FIGS))
	touch $@

SPECIAL =  real basis stick_poster gun_poster
FIGURES_MADE = $(patsubst %, figs/%.pdf, $(SPECIAL)) figs/authors.jpg sim_figs

poster.pdf: poster.tex ${FIGURES_MADE}
	pdflatex poster

###---------------
### Local Variables:
### mode: makefile
### End:

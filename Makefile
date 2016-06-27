# For making documents that describe fitting an eos.  Now the list of
# documents is:

# 1. notes.pdf
#    make clean; time make -j8 notes.pdf" reports 0m59.368s on a
#    4-core 8-thread intel E5-1620 v2 @ 3.70GHz cpu

# Use python2 rather than python3 because debian doesn't have a
# python3 cvxopt version.

# FIGURES is a list of all figures required for notes.pdf.  See these docs:
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
# https://www.gnu.org/software/make/manual/html_node/Text-Functions.html
# Type "make test" to see the list of figures

# Use system default python
PYTHON = /usr/bin/python

# The following lets this code run on moonlight at LANL:
#PYTHON = eval `/usr/bin/modulecmd bash purge`; eval			\
#`/usr/bin/modulecmd bash load python/2.7-anaconda-2.1.0`; export	\
#PYTHONPATH=/usr/projects/fvs/lib/python/; python

FIG_ROOT = CJ_stick opt_stick info_stick big_d vt_gun C_gun BC_gun \
info_gun fve_gun basis

FIGURES = $(patsubst %, figs/%.pdf, ${FIG_ROOT})

CODE = plot.py fit.py eos.py gun.py

notes.pdf: notes.aux notes.bbl ${FIGURES}
	pdflatex notes
notes.aux: notes.tex notes.bbl ${FIGURES}
	pdflatex notes
notes.bbl: local.bib notes.tex ${FIGURES}
	pdflatex notes
	bibtex notes

figs/basis.pdf: basis.py
	mkdir -p figs
	${PYTHON} basis.py $@	
figs/%.pdf: ${CODE}
	mkdir -p figs
	${PYTHON} plot.py --$* $*.pdf

test:
	@echo FIGURES = ${FIGURES}
clean:
	rm -rf figs *.pdf *.pyc *.log *.aux *.bbl *.blg

###---------------
### Local Variables:
### eval: (makefile-mode)
### eval: (setq ispell-personal-dictionary "./localdict")
### End:
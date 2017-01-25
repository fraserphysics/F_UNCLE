# Depends on files built in ../poster
PYTHON = export PYTHONPATH=../../F_UNCLE/;  python
# slides.pdf is the default target
slides.pdf:

surf.pdf: surf.py
	python surf.py $@
figs: ../poster/figs/basis.pdf
	ln -s ../poster/figs $@
static_figs: ../poster/static_figs
	ln -s ../poster/static_figs $@
slides.aux: slides.tex figs static_figs surf.pdf
	pdflatex slides
slides.pdf: slides.aux
	pdflatex slides

###---------------
### Local Variables:
### mode: makefile
### End:

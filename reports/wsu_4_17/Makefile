# Deived from ../ut_2_17 ../poster and ~/projects/eos
slides.pdf:

FIGS = $(addsuffix .pdf, hixson lognom surf $(addprefix figs/, gun_poster gun_tv gun_fisher))
slides.aux: slides.tex figs static_figs surf.pdf hixson.pdf lognom.pdf
	pdflatex slides
slides.pdf: slides.aux
	pdflatex slides

lognom.pdf: ; # Gets made as side effect of rule for hixson.pdf
hixson.pdf: shaw.py
	python2.7 shaw.py hixson.pdf lognom.pdf
clean:
	rm -f *.pyc ui*.py
###---------------
### Local Variables:
### mode: makefile
### End:

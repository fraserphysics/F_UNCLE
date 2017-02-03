# Depends on files built in ../poster
PYTHON = export PYTHONPATH=../../F_UNCLE/;  python
# slides.pdf is the default target
# Use the following to make an environment with PyQt4
# conda create --name qtmayavi scipy pyqt=4 pyface mayavi=4.4.0 python=2.7 ipython
# 1m12.065s
# conda remove --name qtmayavi --all
# 0m51.317s
# 
# In 2016, I used pyside.  This makes an environment with pyside
# conda create --name qtmayavi scipy pyside pyface mayavi python=2.7
slides.pdf:

surf.pdf: surf.py
	python surf.py $@
figs: ../poster/figs/basis.pdf
	ln -sf ../poster/figs $@
static_figs: ../poster/static_figs
	ln -sf ../poster/static_figs $@
slides.aux: slides.tex figs static_figs surf.pdf
	pdflatex slides
slides.pdf: slides.aux
	pdflatex slides

# Translate files made with qtcreator or designer-qt4 to python
ui_PVE_control.py: PVE_control.ui
	pyuic4 $< > ui_temp.py
	mv ui_temp.py $@
ui_eos_qt.py: eos_qt.ui
	pyuic4 $< > ui_temp.py
	patch ui_temp.py patch_eos_ui
	mv ui_temp.py $@
ui_no_mayavi_qt.py: eos_qt.ui patch_no_mayavi
	pyuic4 $< > ui_temp.py
	patch ui_temp.py patch_no_mayavi
	mv ui_temp.py $@
#	pyside-uic $< > $@

test_gui: ui_eos_qt.py ui_no_mayavi_qt.py ui_PVE_control.py
	#python no_mayavi.py
	python gui_eos.py
clean:
	rm -f *.pyc ui*.py
###---------------
### Local Variables:
### mode: makefile
### End:

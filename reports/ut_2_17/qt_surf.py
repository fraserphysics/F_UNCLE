''' qt_surf.py derived from Enthought example qt_embedding.py
'''

import mayavi.mlab
import traits.api
import traitsui.api # Doing this import creates a QApplication
import mayavi.core.ui.api
class Visualization(traits.api.HasTraits):
    # Scene variable
    scene = traits.api.Instance(mayavi.core.ui.api.MlabSceneModel, ())
    # The panel layout
    view = traitsui.api.View(
        traitsui.api.Item(
            'scene', editor=mayavi.core.ui.api.SceneEditor(),resizable=True,
            show_label=False),
        resizable=True)
    @traits.api.on_trait_change('scene.activated')
    def create_pipeline(self):
        ''' Put frame/axes around surface plot
        '''
        mayavi.mlab.outline()
        mayavi.mlab.axes(ranges=self.ranges,xlabel='P',ylabel='v',zlabel='E')
    def __init__(self # Visualization
        ):
        """ Calculate three 2-d arrays of values to describe EOS
        surface and put the surface into self.scene
        """
        import numpy as np
        import eos
        EOS = eos.ideal()
        traits.api.HasTraits.__init__(self)
        P, v = np.mgrid[1e10:4e10:20j, 1e-6:4e-6:20j]
        P = P.T
        v = v.T
        E = np.empty(v.shape)
        n_i,n_j = v.shape
        for i in range(n_i):
            for j in range(n_j):
                E[i,j] = EOS.Pv2E(P[i,j],v[i,j])
        self.ranges = []
        for a in (P,v,E):
            self.ranges += [a.min(),a.max()]
        scale = lambda z: (z-z.min())/(z.max()-z.min())
        mesh = mayavi.mlab.mesh(
            scale(P), scale(v), scale(E), figure=self.scene.mayavi_scene)
        self.flag = False
#-----------------------------------------------------------------------------
# The QWidget containing the visualization
from PyQt4.QtGui import QWidget, QVBoxLayout
class MayaviQWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        self.visualization = Visualization()
        # The edit_traits call generates the widget to embed.
        self.ui = self.visualization.edit_traits(
            parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

#---------------
# Local Variables:
# eval: (python-mode)
# End:

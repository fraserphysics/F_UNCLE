import sys
import os
import time
import unittest

#from ..PhysicsModel import GaussianModel
from test_PhysicsModel import SimpleModel
from test_Simulation import SimpleSimulation

class BigModel(SimpleModel):
    """Simplified physics model with many DOF
    """
    def __init__(self, prior, name="Simplified physics model"):
        """Create dummy physics model
        """
        def_opts = {'nDOF': [int, 10, None, None, '', 'Number of DOF'],
                    'sigma': [float, 1.0, None, None, '', 'Variance']
                    }

        PhysicsModel.__init__(self, None, name=name, def_opts=def_opts)
        self.dof = np.array(prior, dtype=np.float64)
        self.prior = copy.deepcopy(self)

    def _on_str(self):
        """Prints the dof
        """

        out_str = "dof\n"
        for i, value in enumerate(self.dof):
            out_str += '{:d} {:f}'.format(i, value)
        # end

        return out_str
    def get_sigma(self):
        """Get variance
        """

        return np.diag(self.get_dof() *
                       self.get_option('sigma'))

    def get_dof(self):
        """Get dofs
        """

        return self.dof

    def get_scaling(self):
        return np.diag(self.prior.get_dof())

    def update_dof(self, new_dof):
        """Gives a new instance with updated dof
        """

        new_model = copy.deepcopy(self)
        new_model.dof = copy.deepcopy(np.array(new_dof, np.float64))
        return new_model

    def shape(self):
        """Return the shape
        """
        return self.get_option('nDOF')

    def __call__(self, x):
        """Run the model
        """
       
        return x * self.dof



def test_multi_solve(models, sim):
    
    dofs = []
    for i in range(4):
        dofs.append(models['simp'] + i)
    # end
        
    results = sim.multi_solve(models, ['simp',], dofs)

    for i, res in enumerate(results):
        npt.assert_array_equal(
            res[1][0],
            np.arange(10) * (np.arange(10) + i + 1)
        )

    def test_sens(models, sim):
        pass
    
if __name__ == '__main__':
    
    models ={'simp': BigModel(np.arange(10) + 1)}
    sim = SimpleSimulation

    test_multi_solve(models, sim)


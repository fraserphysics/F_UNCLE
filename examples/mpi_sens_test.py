import sys
import os
import time
import copy
import unittest

import numpy as np
import numpy.testing as npt


from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
myrank = mpi_comm.Get_rank()



#from ..PhysicsModel import GaussianModel
from F_UNCLE.Utils.test.test_PhysicsModel import SimpleModel
from F_UNCLE.Utils.test.test_Simulation import SimpleSimulation
from F_UNCLE.Utils.PhysicsModel import PhysicsModel
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
        dofs.append(models['simp'].get_dof() + i)
    # end
        
    results = sim.multi_solve(models, ['simp',], dofs)

    for i, res in enumerate(results):
        npt.assert_array_equal(
            res[1][0],
            np.arange(10) * (np.arange(10) + i + 1)
        )

def test_multi_solve_pll(models, sim):
    
    dofs = []
    for i in range(20):
        dofs.append(models['simp'].get_dof() + i)
    # end
        
    results = sim.multi_solve_mpi(models, ['simp',], dofs, verb=False)


    # for i in results:
    #     print('rank {:d} item{:d} {:}'.format(myrank, i, results[i][1][0]))

    for i, res in enumerate(results):        
        npt.assert_array_equal(
            res[1][0],
            np.arange(10) * (np.arange(10) + i + 1),
            err_msg='Error in {:d} dof set'.format(i)
        )
        
def test_sens_pll(models, sim):

    results = sim.get_sens_mpi(models, ['simp',])
    #print(results)

    resp_mat = np.zeros((10,10), dtype=np.float64)
    inp_mat = np.zeros((10,10), dtype=np.float64)   
    for i in range(models['simp'].shape()):
        dof_new = np.arange(10, dtype=np.float64) + 1
        delta = np.zeros((10,))
        delta[i] = 0.02 * dof_new[i]
        dof_new += delta
        #print(dof_new)
        resp_mat[:, i] = np.arange(10) * dof_new\
                         - np.arange(10) * (np.arange(10) + 1)
        inp_mat[:, i] = delta
    
    sens_matrix = np.linalg.lstsq(inp_mat, resp_mat.T)[0].T        

    # print(resp_mat)
    # print(inp_mat)    
    # print(sens_matrix)

    for i in range(models['simp'].shape()):        
        npt.assert_array_almost_equal(
            results[i , :],
            sens_matrix[i, :],
            err_msg='Error in {:d} dof sens'.format(i)
        )

def test_sens(models, sim):

    results = sim.get_sens(models, ['simp',])
    #print(results)

    resp_mat = np.zeros((10,10), dtype=np.float64)
    inp_mat = np.zeros((10,10), dtype=np.float64)   
    for i in range(models['simp'].shape()):
        dof_new = np.arange(10, dtype=np.float64) + 1
        delta = np.zeros((10,))
        delta[i] = 0.02 * dof_new[i]
        dof_new += delta
        #print(dof_new)
        resp_mat[:, i] = np.arange(10) * dof_new\
                         - np.arange(10) * (np.arange(10) + 1)
        inp_mat[:, i] = delta
    
    sens_matrix = np.linalg.lstsq(inp_mat, resp_mat.T)[0].T        

    # print(resp_mat)
    # print(inp_mat)    
    # print(sens_matrix)

    for i in range(models['simp'].shape()):        
        npt.assert_array_almost_equal(
            results[i , :],
            sens_matrix[i, :],
            err_msg='Error in {:d} dof sens'.format(i)
        )
        
    
if __name__ == '__main__':
    
    models ={'simp': BigModel(np.arange(10, dtype=np.float64) + 1)}
    sim = SimpleSimulation()

    test_multi_solve(models, sim)
    test_multi_solve_pll(models, sim)    
    test_multi_solve_pll(models, sim)
    test_multi_solve_pll(models, sim)    
    test_sens(models, sim)
    test_sens_pll(models, sim)
    test_sens_pll(models, sim)
    test_sens_pll(models, sim)        

"""Uses pyOpt for aposteiori maximization


History
-------

13-06-16 -> Initial class creation
"""
import sys
import os
import pdb
import copy
sys.path.append(os.path.abspath('./../../'))

import numpy as np
from pyOpt import Optimization
from pyOpt.pyCONMIN import CONMIN

from F_UNCLE.Opt.Bayesian import Bayesian as OldBayesian


class BayesPyOpt(OldBayesian):
    def _local_opt(self, sims, model, inital_data):
        """Solves for the maximum aposteori likelyhood
        
        see :py:meth:`F_UNLCE.Opt.Bayesian._local_opt`
        """

        def objfun(x, param=None):
            """Objective function                                   
            """

            print "In the objective function"
            sims = copy.deepcopy(param['sims'])
            model = copy.deepcopy(param['model'])
            bayes = copy.deepcopy(param['bayes'])

            # print model.shape()
            # print x.shape
            # print x[50:]
            bayes.model.set_dof(x[:50])

            initial_data = []
            for sim, exp in sims:
                sim.update(model=bayes.model)
                initial_data.append(sim())
            #end

            log_like = 0
            log_like += bayes.model_log_like()
            log_like += bayes.sim_log_like(initial_data)
            log_like *= 1

            m_dof = bayes.shape()[1]
            model_indep = model.get_t()
            
            g = np.zeros(3*m_dof-1)
            g[:m_dof] = model.derivative(n=1)(model_indep[4:])
            g[m_dof:2*m_dof] = -1 * model.get_dof()
            g[2*m_dof:] = model.get_dof()[1:] - model.get_dof()[:-1]
                                  
            return float(log_like), g, False

        m_dof = model.shape()
        initial_dof = model.get_dof()
        opt_prob = Optimization('Maximum a Posteriori Optimization', objfun)

        opt_prob.addObj('A Posteriori Likelyhood')
        
        for i in xrange(m_dof[0]):
            opt_prob.addVar('Knot {:d}'.format(i), 'c', lower = 0.0,
                            upper = 1E20, value = initial_dof[i] )
        #end
        
        opt_prob.addConGroup('Convexity', m_dof[0])
        opt_prob.addConGroup('Positive', m_dof[0])        
        opt_prob.addConGroup('Monotonicity', m_dof[0]-1)

        print opt_prob

        optimizer = CONMIN()

        params = {
            'model':model,
            'sims':sims,
            'bayes':self,
            }

        f_star, x_star, inform = optimizer(opt_prob, sens_step = 5E7, param = params)

        print opt_prob.solution(0)

        return x_star[:50]
if __name__ == '__main__':
    from F_UNCLE.Experiments.GunModel import Gun
    from F_UNCLE.Experiments.Stick import Stick
    from F_UNCLE.Models.Isentrope import EOSModel, EOSBump

    # Initial estimate of prior functional form
    init_prior = np.vectorize(lambda v: 2.56e9 / v**3)

    # Create the model and *true* EOS
    eos_model = EOSModel(init_prior)
    eos_true = EOSBump()

    # Create the objects to generate simulations and pseudo experimental data
    gun_experiment = Gun(eos_true)
    gun_simulation = Gun(eos_model)
    gun_prior_sim =gun_simulation()


    analysis = BayesPyOpt(simulations = [(gun_simulation, gun_experiment)],
                        model = eos_model, constrain = True, precondition = False)

    # Run the analysis
    best_eos, history = analysis()


        
            


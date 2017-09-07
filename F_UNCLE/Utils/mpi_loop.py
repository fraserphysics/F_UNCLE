#!/usr/local/bin/python
"""

mpi_loop

Abstract paralellized loop. Used to apply arbitrary inputs to arbitrary
functions
- Uses Static Process Management


Developers
----------
- Dr. S. Andrews

History
-------
1.0 - Initial class creation

"""

import numpy as np
import time

try:
    from mpi4py import MPI
except Exception as inst:
    pass
# end


def pll_loop(x, func, shape, comm=None, verb=False, *args, **kwargs):
    """
    Uses MPI to evaluate `func' for each value of x and returns a dictionary of
    the results
    - Uses Static Process Management (SPM)

    Args:
        x(list): Inputs to the function, can be a list of values, lists or
                 dictionarys.
        func(function): The function to evaluate.

    Keyword Args:
        comm(mpi4py communicator): If none, uses the whole set of MPI processprs
            else it should be a valid MPI group to use.

    Return:
         (dict): Dictionary of the function outputs with the same indicies as x.

    """
    # print('Entering the MPI loop')
    if comm is None:
        if verb: print('Generating new communicator')
        try:
            comm = MPI.COMM_WORLD
        except:
            raise(ImportError("Could not import MPI for py"))
        # end
    else:
        pass
    # end
    nproc = comm.Get_size()
    myrank = comm.Get_rank()
    if verb: print("Hello from rank %d of %d" % (myrank, nproc))
    Barrier = comm.barrier
    Send = comm.send
    Recv = comm.recv
    Bcast = comm.bcast

    Barrier()
    x = Bcast(x, root=0)
    myxval = range(myrank, len(x), nproc)
    out_data = {}

    # The actuall paralell loop
    send_buf = []  # np.empty((shape, len(myxval)), dtype=np.float64)
    for j, i in enumerate(myxval):
        if verb: print('evaluating item {:d} on rank {:d}\nvariables are {:}'
                       .format(i, myrank, x[i]))
        send_buf.append(func(x[i], *args, **kwargs))
        # if verb: print('rank {:d} finished item {:d}\n results are {:}'
        #                .format(myrank, i, send_buf[-1][1][0]))
    # end

    # Send and receive the data
    if myrank != 0:
        if verb: print('sending from rank {:d} to rank {:d}'.format(myrank, 0))
        req = Send(send_buf, dest=0, tag=1)
    else:
        for j, i in enumerate(myxval):
            out_data[i] = send_buf[j]
        # end
    # end

    if myrank == 0:
        if verb: print('rank 0 begining to recieve')
        for proc in range(1, nproc):
            proc_xval = range(proc, len(x), nproc)
            rec_buff = np.empty((shape, len(proc_xval)), dtype=np.float64)
            try:
                rec_buff = Recv(source=proc, tag=1)
                if verb: print('recieving from rank {:d} on rank {:d}'
                               .format(proc, myrank))
                for j, i in enumerate(proc_xval):
                    out_data[i] = rec_buff[j]
                # end
            except MPI.Exception as inst:
                print("rank %d failed recieving from proc %d" % (myrank, proc))
                print(type(inst))
                raise inst
        # end
    # end

    out_data = Bcast(out_data, root=0)
    Barrier()

    return out_data
# end


def expensive_func(x, *args, **kwargs):
    """
    A function to evaluate to test the paralell abilities
    """
    fact = x
    for i in range(x - 1):
        fact *= (x - (i + 1))
    # end
    time.sleep(1)
    return fact


def complex_io(x, p={}, *args, **kwargs):
    """
    A function to test the IO capabilities of the function
    """
    var1 = x[0]
    var2 = x[1]

    kw1 = p['kw1']

    # print "var1 ", var1
    # print "var2 ", var2
    # print "kw1 ", kw1

    out = {}
    out['out1'] = str(var1 * var2)
    out['out2'] = var1 + var2
    out['out3'] = var1 * 1.0 / var2
#    time.sleep(10)
    return out

if __name__ == "__main__":
    # from mpi4py import MPI

    # # Test 1
    # # ------
    # # > Test the speedup obtained from the paralell function

    # import time
    # x = range(2,4, 1)
    # to = time.time()
    # y = pll_loop(x, expensive_func, shape=1)

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     print("Time taken ", time.time()-to)
    # #end

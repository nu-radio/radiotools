

def get_GE_preexecution(jobname=None, rundir=None,
                        mailflags='beas',
                        ncores=1,
                        mpi=False):
    from io import StringIO

    output = StringIO()

    output.write("#!/bin/bash\n")
    output.write("#$ -N {}\n".format(jobname))
    output.write("#$ -j y\n")
    output.write("#$ -V\n")
    output.write("#$ -q grb,grb64\n")
    if(mailflags != ""):
        output.write("#$ -m {}\n".format(mailflags))
    output.write("#$ -o {}\n".format(rundir))
    if(mpi):
        output.write("#$ -pe mpi {}\n".format(ncores))
        output.write("#$ -R y\n")
    return output.getvalue()

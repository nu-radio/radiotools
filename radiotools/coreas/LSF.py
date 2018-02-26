

def get_LSF_preexecution(jobname=None, outputlog=None, runtime=1, memory=1850.,
                         ncores=12, projectid=None,
                         mpi=True, hpcwork=True, email=None):
    import StringIO
    import numpy as np
    import copy

    output = StringIO.StringIO()
    output.write("#!/usr/bin/env zsh\n")

    output.write("### Job name\n")
    output.write("#BSUB -J %s\n" % jobname)

    output.write("### File / path where STDOUT & STDERR will be written\n")
    output.write("###    %J is the job ID, %I is the array ID\n")
    if outputlog is None:
        output.write("#BSUB -o parcotest.%J.%I\n")
    else:
        output.write("#BSUB -o %s\n" % outputlog)

    if email is not None:
        output.write("#BSUB -B\n")
        output.write("#BSUB -N\n")
        output.write("#BSUB -u %s\n" % email)

    if projectid is not None:
        output.write("#BSUB -P %s\n" % projectid)

    output.write("#BSUB -W %i:%02i\n" % (int(np.floor(runtime)), (runtime % 1) * 60))

    output.write("#BSUB -M %.0f\n" % memory)

    output.write("#BSUB -S 600 # increase stack size\n")

    output.write("#BSUB -n %i\n" % ncores)
    output.write("#BSUB -x\n")

    if mpi:
        output.write("#BSUB -m mpi-s\n")
        output.write("#BSUB -a openmpi\n")
        output.write("#BSUB -R \"span[ptile=12]\"\n")

    if hpcwork:
        output.write("#BSUB -R \"select[hpcwork]\"\n")

    output.write("module load python/2.7.11\n")

    return output.getvalue()

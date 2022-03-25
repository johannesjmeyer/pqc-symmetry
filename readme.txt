We are trying to showcase the advantage of exploiting symmetries of datasets when training parametrized quantum circuits.

This code in particular implements gradient descent to train some PQCs to classify "tic-tac-toe" games based on whether X, O or neither has won.

The code is written in python (3) and uses libraries available through pip (see requirements.txt). 

A typical call to a single training run (carried out by run_ttt.py) looks like this:

python3 run_ttt.py -s "true" -n "$steps" -p "6" -l "tcemoid" -f "output/depth_7" -ss "0.008" -sr "true" -re "$depth" -ep "true" -epn $epochs

with options:
    -s "true"           => use symmetrized circuits
    -n "$steps"         => number of parameters update per training run
    -p "6"              => number of points to compute gradient (more steps = longer training)
    -l "tcemoid"        => sets geometry of the PQC
    -f "output/depth_7" => relative path to save output files
    -ss "0.008"         => step size for gradient descent
    -sr "true"          => points for grad computation are randomly sampled
    -re "$depth"        => number of repetitions of the basic PQC to construct full PQC
    -ep "true"          => if true, each data point is seen more than once (use epochs)
    -epn $epochs        => number of epochs



SLURM scripts:

    1) ttt_slurm_script.sh

        runs the same training 20 times for both symmetric and non-symmetric circuits (for smooth plots...), 5 trainings should run in parallel, allocate 4 cores each (numpy)

    2) ttt_slurm_test.sh 

        same as 1) but shorter training, should take <5 min





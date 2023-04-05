#   This is the most basic QSUB file needed for this cluster.
#   Further examples can be found under /share/apps/examples
#   Most software is NOT in your PATH but under /share/apps
#
#   For further info please read http://hpc.cs.ucl.ac.uk
#   For cluster help email cluster-support@cs.ucl.ac.uk

#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=12:00:00
#$ -l gpu=true,gpu_type=!(gtx1080ti|rtx2080ti|titanxp|titanx)

#$ -S /bin/bash
#$ -wd /cluster/project2/ifo/
#$ -j y # merge stdout and stderr
#$ -N adlg

# no ### $ -t 1-100

cd /cluster/project2/ifo/lfo/comp0197-group-project
echo "pwd:"
pwd
echo "Task ID: ${SGE_TASK_ID}"

/cluster/project2/ifo/anaconda3/bin/conda run -n comp0197-group-project python run.py

### ${SGE_TASK_ID}

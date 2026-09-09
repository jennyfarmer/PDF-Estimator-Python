#!/bin/bash

#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

module load python/3.12.3
source /users/jfarmer6/pdfePython/pdfePy/venv/bin/activate
cd /users/jfarmer6/pdfePython/pdfePy/examples

arrayCount=0
for distribution in $DISTRIBUTIONS
do
	for seed in $SEEDS
	do
		for samples in $SAMPLE_SIZES
		do
			distArray[$arrayCount]=$distribution
			seedArray[$arrayCount]=$seed
			samplesArray[$arrayCount]=$samples
			arrayCount=$((arrayCount +1))
		done
	done
done

python run_examples_slurm.py \
                    ${distArray[${SLURM_ARRAY_TASK_ID}]} \
                    ${samplesArray[${SLURM_ARRAY_TASK_ID}]} \
                    ${seedArray[${SLURM_ARRAY_TASK_ID}]} \
                    $WRITE_FILE \
                    $EVALUATE_KDE \
                    $PLOT_PDFS \
                    $VERBOSE

find slurm_logs -type f -empty -delete
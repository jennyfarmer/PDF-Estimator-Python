distributions=({2..2})
seeds=({0..10})
sample_sizes=(50000)
#50000)

plot_pdfs=false
evaluate_kde=false
verbose=true
write_file=true

n_jobs=$((${#distributions[@]} * ${#seeds[@]} * ${#sample_sizes[@]}))

printf "Distributions: %s\n" "${distributions[*]}"
printf "Seeds:         %s\n" "${seeds[*]}"
printf "Sample sizes:  %s\n" "${sample_sizes[*]}"
printf "Total jobs:    %d\n" "$n_jobs"

export_vars="ALL"
export_vars+=",DISTRIBUTIONS=${distributions[*]}"
export_vars+=",SEEDS=${seeds[*]}"
export_vars+=",SAMPLE_SIZES=${sample_sizes[*]}"
export_vars+=",WRITE_FILE=$write_file"
export_vars+=",EVALUATE_KDE=$evaluate_kde"
export_vars+=",PLOT_PDFS=$plot_pdfs"
export_vars+=",VERBOSE=$verbose"

if [ "$verbose" = true ]; then
    output_file="slurm_logs/slurm_%A_%a.out"
else
    output_file="/dev/null"
fi

sbatch \
    --array=0-$((n_jobs - 1)) \
    --output=$output_file \
    --error=slurm_logs/slurm_%A_%a.err \
    --export="$export_vars" \
    run_examples.slurm


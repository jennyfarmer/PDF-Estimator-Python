import compare_estimators

seeds = range(1)
sample_sizes = [5000]
distribution_ids = [0, 1, 3]

plot_pdfs = True
evaluate_kde = False
verbose = True
write_file = False


def main() -> None:

    for distribution_id in distribution_ids:
        for seed in seeds:
            for sample_size in sample_sizes:
                compare_estimators.compare(
                    distribution_id=distribution_id,
                    n_samples=sample_size,
                    seed=seed,
                    write_file=write_file,
                    evaluate_kde=evaluate_kde,
                    plot_pdfs=plot_pdfs,
                    verbose=verbose,
                )

if __name__ == "__main__":
    main()

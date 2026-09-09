import sys
import compare_estimators


def main():
    distribution_id = int(sys.argv[1])
    sample_size = int(sys.argv[2])
    seed = int(sys.argv[3])
    write_file = sys.argv[4].lower() == "true"
    evaluate_kde = sys.argv[5].lower() == "true"
    plot_pdfs = sys.argv[6].lower() == "true"
    verbose = sys.argv[7].lower() == "true"

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
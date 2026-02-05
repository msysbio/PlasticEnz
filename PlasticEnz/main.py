#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from time import sleep, time
from tqdm import tqdm

def display_logo():
    """Display the PlasticEnz logo and citations."""
    logo = """
       ___ _           _   _        __          
      / _ | | __ _ ___| |_(_) ___  /___ __  ____
     / /_)| |/ _` / __| __| |/ __|/_\| '_ \|_  /
    / ___/| | (_| \__ | |_| | (__//__| | | |/ / 
    \/    |_|\__,_|___/\__|_|\___\__/|_| |_/___|
                                                
    
    PlasticEnz - Plastic-Degrading Enzyme Detection Pipeline
    
    """
    print(logo)
    print("\nPlease remember to cite these references when using PlasticEnz:")
    print("- Prodigal: Hyatt et al., 2010. BMC Bioinformatics. DOI: 10.1186/1471-2105-11-119")
    print("- HMMER: Eddy, 2011. PLoS Comput Biol. DOI: 10.1371/journal.pcbi.1002195")
    print("- DIAMOND: Buchfink et al., 2015. Nat Methods. DOI: 10.1038/nmeth.3176")
    print("- Bowtie2: Langmead & Salzberg, 2012. Nat Methods. DOI: 10.1038/nmeth.1923")
    print("- Samtools: Danecek et al., 2021. Gigascience. DOI: 10.1093/gigascience/giab008")
    print("- ProtTrans: Elnaggar et al., 2022. IEEE TPAMI. DOI: 10.1109/TPAMI.2021.3095381")
    print("\n")


def simulated_loading_bar(task_name, duration=3):
    """Simulate a loading bar for a task."""
    print(f"\n{task_name} is starting. Please wait...")
    for _ in tqdm(range(duration), desc=f"{task_name} in progress", unit="steps"):
        sleep(1)
    print(f"{task_name} completed.\n")


def parse_args():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="PlasticEnz: A tool for detecting plastic-degrading enzymes from sequence data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files
    input_group = parser.add_argument_group('Input Files')
    input_group.add_argument("-c", "--contigs", type=str, help="Path to contigs file (FASTA).")
    input_group.add_argument("-1", "--reads_forward", type=str, help="Path to forward reads file (FASTQ).")
    input_group.add_argument("-2", "--reads_reverse", type=str, help="Path to reverse reads file (FASTQ).")
    input_group.add_argument("-p", "--proteins", type=str, help="Path to protein file (FASTA).")
    input_group.add_argument("-g", "--genome", type=str, help="Path to genome or MAG file (FASTA).")

    # Analysis parameters
    analysis_group = parser.add_argument_group('Analysis Parameters')
    analysis_group.add_argument("--polymer", type=str, default=None,
                                help="Polymer(s) to screen for: LDPE,PBSA,PBS,PCL,PES,PHBV,PLA,P3HP,PBAT,PEA,PET,PHA,PHB. Use 'all' for all available.")
    analysis_group.add_argument("--outdir", type=str, default=None,
                                help="Output directory.")
    analysis_group.add_argument("--cores", type=int, default=1,
                                help="Number of CPU cores to use.")

    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument("--use_gpu", action="store_true",
                            help="Attempt to use GPU for accelerated computations.")
    perf_group.add_argument("--sensitive", action="store_true",
                            help="Use neural network model (nn_model.pkl) for sensitive predictions.")

    # Thresholds
    threshold_group = parser.add_argument_group('Search Thresholds')
    threshold_group.add_argument("--evalue_hmmer", type=float, default=1e-5,
                                 help="E-value threshold for HMMER search.")
    threshold_group.add_argument("--bitscore_hmmer", type=float, default=20,
                                 help="Bitscore value for HMMER search.")
    threshold_group.add_argument("--evalue_diamond", type=float, default=1e-5,
                                 help="E-value threshold for DIAMOND search.")
    threshold_group.add_argument("--bitscore_diamond", type=float, default=20,
                                 help="Minimum alignment quality for DIAMOND search.")

    # Test mode
    parser.add_argument("--test", action="store_true",
                        help="Run the tool with a predefined test dataset.")

    return parser


# Configuration Functions

def configure_test_mode(args, test_data_dir):
    """Configure arguments for test mode."""
    print("\n" + "-" * 70)
    print("Running in TEST MODE")
    print("-" * 70)

    args.genome = os.path.join(test_data_dir, "Bacterioplankton_PLA.117.fa")
    args.reads_forward = ",".join([
        os.path.join(test_data_dir, "S1_forward_1.fastq"),
        os.path.join(test_data_dir, "S2_forward_2.fastq"),
        os.path.join(test_data_dir, "S3_forward_3.fastq"),
    ])
    args.reads_reverse = ",".join([
        os.path.join(test_data_dir, "S1_reverse_1.fastq"),
        os.path.join(test_data_dir, "S2_reverse_2.fastq"),
        os.path.join(test_data_dir, "S3_reverse_3.fastq"),
    ])
    args.polymer = "PET,PLA"

    print("\nTest Configuration:")
    print("  Genome: {}".format(os.path.basename(args.genome)))
    print("  Forward reads: 3 samples")
    print("  Reverse reads: 3 samples")
    print("  Polymers: {}".format(args.polymer))
    print("  Output directory: {}".format(args.outdir))

    # Check for SignalP availability
    try:
        import shutil
        if shutil.which("signalp6"):
            args.signalp = True
            args.signalp_mode = "fast"
            args.signalp_organism = "auto"
            args.signalp_batch_size = 8
            if args.signalp_outdir is None:
                args.signalp_outdir = os.path.join(args.outdir, "signalp_test")
            print("  SignalP: enabled (fast, auto)")
        else:
            print("  SignalP: skipped (not found in PATH)")
    except Exception as e:
        print("  SignalP: skipped ({})".format(e))

    print("-" * 70 + "\n")


def setup_logging(outdir):
    """Configure logging for the pipeline."""
    logger = logging.getLogger("PlasticEnz")
    logger.setLevel(logging.INFO)

    main_log_file = os.path.join(outdir, "main.log")
    handler = logging.FileHandler(main_log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

#Pipeline steps

def run_gene_prediction(args, logger):
    """Run Prodigal for gene prediction."""
    from PlasticEnz.modules.run_prodigal import run_prodigal

    if args.genome:
        print("\n" + "-" * 70)
        print("STEP 1: Gene Prediction (Prodigal)")
        print("-" * 70)
        print("Input: {}".format(os.path.basename(args.genome)))
        print("Running Prodigal (this may take a few minutes)...")
        sys.stdout.flush()

        try:
            protein_out, gene_out, _ = run_prodigal(
                args.genome, args.outdir, cores=args.cores, is_genome=True
            )
            print("Gene prediction complete")
            print("  Proteins: {}".format(protein_out))
            print("  Genes: {}\n".format(gene_out))
            return protein_out, gene_out
        except Exception as e:
            logger.error("Error running Prodigal: {}".format(e))
            sys.exit("ERROR: Prodigal failed on genome/MAG file: {}".format(e))

    elif args.contigs:
        print("\n" + "-" * 70)
        print("STEP 1: Gene Prediction (Prodigal)")
        print("-" * 70)
        print("Input: {}".format(os.path.basename(args.contigs)))
        print("Running Prodigal (this may take a few minutes)...")
        sys.stdout.flush()

        try:
            protein_out, gene_out, _ = run_prodigal(
                args.contigs, args.outdir, cores=args.cores, is_genome=False
            )
            print("Gene prediction complete")
            print("  Proteins: {}".format(protein_out))
            print("  Genes: {}\n".format(gene_out))
            return protein_out, gene_out
        except Exception as e:
            logger.error("Error running Prodigal: {}".format(e))
            sys.exit("ERROR: Prodigal failed on contigs file: {}".format(e))

    else:
        print("\n" + "-" * 70)
        print("STEP 1: Using Provided Proteins")
        print("-" * 70)
        print("Input: {}\n".format(os.path.basename(args.proteins)))
        return args.proteins, None


def run_homology_searches(protein_file, valid_polymers, args, logger):
    """Run HMMER and DIAMOND searches for each polymer."""
    from PlasticEnz.modules.run_hmmer import run_hmmer
    from PlasticEnz.modules.run_diamond import run_diamond

    print("-" * 70)
    print("STEP 2: Homology Searches")
    print("-" * 70)
    print("Target polymers: {}".format(", ".join(valid_polymers)))
    print("Running searches for {} polymer(s)\n".format(len(valid_polymers)))
    sys.stdout.flush()

    hmmer_outputs = []
    diamond_outputs = []

    for idx, poly in enumerate(valid_polymers, 1):
        print("[{}/{}] Processing {}:".format(idx, len(valid_polymers), poly))

        # HMMER search
        print("  Running HMMER search...")
        sys.stdout.flush()
        try:
            outs = run_hmmer(
                proteins=protein_file,
                polymer=poly,
                bitscore=args.bitscore_hmmer,
                outdir=args.outdir,
                evalue=args.evalue_hmmer,
                cores=args.cores,
            )
            hmmer_outputs.extend(outs)
            print("  HMMER complete")
        except Exception as e:
            logger.error("Error running HMMER for {}: {}".format(poly, e))
            sys.exit("ERROR: HMMER failed for {}: {}".format(poly, e))

        # DIAMOND search
        print("  Running DIAMOND search...")
        sys.stdout.flush()
        try:
            outs = run_diamond(
                proteins=protein_file,
                polymer=poly,
                outdir=args.outdir,
                evalue=args.evalue_diamond,
                min_score=args.bitscore_diamond,
                cores=args.cores,
            )
            diamond_outputs.extend(outs)
            print("  DIAMOND complete\n")
        except Exception as e:
            logger.error("Error running DIAMOND for {}: {}".format(poly, e))
            sys.exit("ERROR: DIAMOND failed for {}: {}".format(poly, e))

    print("All homology searches complete\n")
    return hmmer_outputs, diamond_outputs


def run_extraction(hmmer_outputs, diamond_outputs, protein_file, final_output_dir, logger):
    """Extract homologues and proteins."""
    from PlasticEnz.modules.run_extract import run_extract

    print("-" * 70)
    print("STEP 3: Extracting Homologues")
    print("-" * 70)
    print("Processing HMMER and DIAMOND results...")
    sys.stdout.flush()

    try:
        summary_file, fasta_file = run_extract(
            hmmer_outputs=hmmer_outputs,
            diamond_outputs=diamond_outputs,
            protein_file=protein_file,
            output_dir=final_output_dir,
        )
        print("Extraction complete")
        print("  Summary: {}".format(summary_file))
        print("  Sequences: {}\n".format(fasta_file))
        return summary_file, fasta_file
    except Exception as e:
        logger.error("Error during extraction: {}".format(e))
        sys.exit("ERROR: Extraction failed: {}".format(e))


def run_read_mapping(args, gene_out, logger):
    """Map reads and calculate abundances."""
    from PlasticEnz.modules.run_mapping import run_mapping

    if not (args.reads_forward and args.reads_reverse):
        print("Skipping read mapping (no reads provided)\n")
        return None

    print("-" * 70)
    print("STEP 4: Read Mapping & Abundance Calculation")
    print("-" * 70)

    forward_list = args.reads_forward.split(",")
    reverse_list = args.reads_reverse.split(",")
    print("Processing {} sample(s)".format(len(forward_list)))
    print("Running Bowtie2 mapping (this may take several minutes)...")
    sys.stdout.flush()

    try:
        abundance_file = run_mapping(
            forward_reads=forward_list,
            reverse_reads=reverse_list,
            genes_file=gene_out,
            outdir=args.outdir,
        )
        print("Read mapping complete")
        print("  Abundance file: {}\n".format(abundance_file))
        return abundance_file
    except Exception as e:
        logger.error("Error during reads mapping: {}".format(e))
        sys.exit("ERROR: Reads mapping failed: {}".format(e))


def run_ml_predictions(fasta_file, summary_file, valid_polymers, args, logger):
    """Run machine learning predictions for PET and PHB."""
    from PlasticEnz.modules.run_prediction import run_predictions

    prediction_polymers = [poly for poly in valid_polymers if poly in {"PET", "PHB"}]

    if not prediction_polymers:
        print("Skipping ML prediction step (PET and PHB not in target polymers)\n")
        return

    print("-" * 70)
    print("STEP 5: Machine Learning Predictions")
    print("-" * 70)
    print("Target polymers: {}".format(", ".join(prediction_polymers)))

    # Select model based on sensitivity flag
    if args.sensitive:
        model_path = os.path.join(os.path.dirname(__file__), "data", "model", "nn_model.pt")
        model_tag = "nn"
        print("Using Neural Network model (sensitive mode)")
    else:
        model_path = os.path.join(os.path.dirname(__file__), "data", "model", "xgb_model.pkl")
        model_tag = "xgb"
        print("Using XGBoost model (default mode)")

    if args.use_gpu:
        print("GPU acceleration: enabled")

    print("Running predictions...")
    sys.stdout.flush()
    logger.info("Running prediction step.")

    try:
        run_predictions(
            fasta_file=fasta_file,
            summary_table=summary_file,
            gpu=args.use_gpu,
            model_path=model_path,
            polymers=prediction_polymers,
            model_tag=model_tag
        )
        print("Predictions complete\n")
    except Exception as e:
        logger.error("Error during prediction: {}".format(e))
        sys.exit("ERROR: Prediction failed: {}".format(e))

#Main:

def main():
    """Main entry point for PlasticEnz pipeline."""
    parser = parse_args()

    # Early check: if no arguments or --help/-h is present, show help and exit
    if len(sys.argv) == 1 or any(arg in sys.argv for arg in ("-h", "--help")):
        display_logo()
        parser.print_help()
        sys.exit(0)

    # Delay importing heavy modules until after help check
    import PlasticEnz
    from PlasticEnz.modules.utility_functions import (
        validate_inputs,
        validate_polymers,
        merge_abundance_with_summary,
    )

    # Define test data directory
    TEST_DATA_DIR = os.path.join(os.path.dirname(PlasticEnz.__file__), "test")

    # Display logo and parse arguments
    display_logo()
    args = parser.parse_args()

    # Configure test mode if enabled
    if args.test:
        if not args.outdir:
            parser.error("ERROR: When using --test mode, you must provide --outdir.")
        configure_test_mode(args, TEST_DATA_DIR)
    elif not args.polymer or not args.outdir:
        parser.error("ERROR: The --polymer and --outdir arguments are required unless using --test.")

    # Validate inputs and polymers
    validate_inputs(args)
    valid_polymers = [poly.strip().upper() for poly in validate_polymers(args.polymer)]

    # Create output directories
    os.makedirs(args.outdir, exist_ok=True)
    final_output_dir = os.path.join(args.outdir, "output")
    os.makedirs(final_output_dir, exist_ok=True)

    # Configure logging
    logger = setup_logging(args.outdir)

    # Start timer
    start_time = time()
    import datetime
    print("Pipeline started at: {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # Run pipeline steps
    protein_file, gene_out = run_gene_prediction(args, logger)

    if protein_file is None:
        sys.exit("ERROR: No valid protein file available for HMMER step.")

    hmmer_outputs, diamond_outputs = run_homology_searches(
        protein_file, valid_polymers, args, logger
    )

    summary_file, fasta_file = run_extraction(
        hmmer_outputs, diamond_outputs, protein_file, final_output_dir, logger
    )

    abundance_file = run_read_mapping(args, gene_out, logger)

    run_ml_predictions(fasta_file, summary_file, valid_polymers, args, logger)

    # Calculate elapsed time
    elapsed_time = time() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Pipeline complete - show summary
    print("\n" + "-" * 70)
    print("PlasticEnz Analysis Complete")
    print("-" * 70)
    print("Output directory: {}".format(final_output_dir))
    print("Log file: {}".format(os.path.join(args.outdir, 'main.log')))
    if hours > 0:
        print("Total runtime: {}h {}m {}s".format(hours, minutes, seconds))
    elif minutes > 0:
        print("Total runtime: {}m {}s".format(minutes, seconds))
    else:
        print("Total runtime: {}s".format(seconds))
    print("-" * 70 + "\n")
    logger.info("PlasticEnz analysis completed successfully.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:39:02 2023

@author: u0145079

Quantify hmms
Input:
    A sorted and indexed BAM file

Requires:
    Samtools

"""
import os
import pandas as pd
import sys
from utilities import check_dependencies,spinning_cursor_task
import threading
import subprocess
import time
import glob

error2="""
 __ __  __  __  __  
|_ |__)|__)/  \|__) 
|__| \ | \ \__/| \  
                                     
"""

def check_args(args):
    if args.gene_counts_file is None and args.bams is None:
        error = "\nError: Niether gene_count nor bams file are specified\n"
        error += "You must provide either the --gene_counts_file or the --bams argument.\n"
        print(error2)
        print(error)
        sys.exit(1)
    if args.gene_counts_file is None and args.bams:
        pass  # do nothing
    if args.gene_counts_file and args.bams is None:
        pass  # do nothing

def quantify_hmm(args):
    check_dependencies('samtools')
    # Locate the hmmersearch output file in the temporary folder
    temp_dir = os.path.join(args.output, "temps")

# Load the read_counts table and sum the second column
    if args.gene_counts_file is None and args.bams:
        bam_dir = args.bams
        bam_files = [os.path.join(bam_dir, bam_file) for bam_file in os.listdir(bam_dir) if bam_file.endswith('.bam')]
    
        gene_counts_file = []
    
        for bam_file in bam_files:
            samtools_command = f"samtools idxstats {bam_file}"
            base = os.path.basename(bam_file).split(".")[0]
            log_file = os.path.abspath(os.path.join(temp_dir, f"{base}_samtools.log"))
            samtools_output = os.path.abspath(os.path.join(temp_dir, f"{base}_samtools_output.tsv"))
            
            task_done = threading.Event()
    
            with open(log_file, 'w') as f_err, open(samtools_output, 'w') as f_out:
                process = subprocess.Popen(samtools_command, shell=True, stdout=f_out, stderr=f_err)
    
            t = threading.Thread(target=spinning_cursor_task, args=(task_done,'samtools'))
            t.start()
    
            while process.poll() is None:
                time.sleep(0.1)
    
            task_done.set()
            t.join()
    
            print("\nsamtools finished running. Logs saved to {}".format(log_file))
    
            if not os.path.exists(samtools_output) or os.stat(samtools_output).st_size == 0:
                print(f"SAMTOOLS FAILED FOR {base} file: For more information check samtools log at: {log_file}")
            else:
                gene_counts_file.append(samtools_output)
                
    elif args.gene_counts_file and args.bams is None:
        gene_counts_file = args.gene_counts_file

    # Initialize an empty list to store abundance data for each file

    abundance_data = []    
    hmmer_output_file = [os.path.join(temp_dir, file) for file in os.listdir(temp_dir) if file.endswith("_HMMER.out")]

    if not hmmer_output_file:
        print(error2)
        print("\nError: No hmm output file found in the temporary folder. Make sure to run translate_search() before quantify_enzymes().")
        sys.exit(1)

    for hmm_file in hmmer_output_file:
        enzyme = os.path.basename(hmm_file).split('_')[0]
        # Load the hmmsearch output table and transform it into pandas
        with open(hmm_file) as f:
            lines = f.readlines()
            # Remove lines that start with '#' or are empty
            rows = [line.split() for line in lines if not line.startswith('#') and line.strip()]
            df = pd.DataFrame(rows)
        
            # Remove columns that contain only '#'
            df = df.loc[:, ~df.eq('#').all()]
            df = df.iloc[:, :10]  # subset first 10 columns
            df.columns = ['#target_name', 'target_accession', 'query_name', 'query_accession', 'E_value', 'score', 'bias', 'exp', 'reg', 'clu']  # column names
            # convert numerical columns to numeric types
            numeric_cols = ['E_value', 'score', 'bias', 'exp', 'reg', 'clu']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)  # convert numerical columns to numeric types
            df['contig'] = df['#target_name'].apply(lambda x: '_'.join(x.split('_')[:2]))  # extract contig information
        
            final_abundance = 0
            total_num_reads = 0
            
            for count_file in gene_counts_file:
                read_counts_table = pd.read_csv(count_file, sep='\t', header=None, names=['contig', 'length', 'num_reads', 'unmapped_reads'])
                total_num_reads += read_counts_table['num_reads'].sum()
                merged_table = pd.merge(df, read_counts_table, on='contig')
                merged_table['rpkm'] = (1e9 * merged_table['num_reads']) / (total_num_reads * merged_table['length'])
                final_abundance = merged_table['rpkm'].sum()
            
                # Calculate average bit-score
                bit = df['score'].mean()
            
                # Append final abundance and bit score for each file to the list
                abundance_data.append([os.path.basename(count_file), final_abundance, bit,enzyme])
            
            # Convert the abundance_data list to a DataFrame
            abundance_df = pd.DataFrame(abundance_data, columns=['File', 'Final Abundance', 'Bit Score', 'Plastic_substrate'])
            
            # Specify the output directory
            output_directory = os.path.join(args.output, 'output')
            
            # Create the output directory if it does not exist
            os.makedirs(output_directory, exist_ok=True)
            
            # Write the DataFrame to a .csv file in the output directory
            output_file = os.path.join(output_directory, 'abundance_output.csv')
            abundance_df.to_csv(output_file, index=False)





import subprocess
import os
import logging

def filter_hmmer_output_inplace(tblout_file,bias_ratio=0.1):
    """Filter HMMER tblout output to remove duplicate hit IDs."""
    temp_file = f"{tblout_file}.tmp"
    seen_ids = set()

    with open(tblout_file, "r") as infile, open(temp_file, "w") as outfile:
            for line in infile:
                if line.startswith("#"):
                    outfile.write(line)
                    continue

                # HMMER --tblout: we only need first 7 fields to get score/bias.
                fields = line.strip().split()
                if len(fields) < 7:
                    continue

                target = fields[0]
                query  = fields[2]
                key = (target, query)
                if key in seen_ids:
                    continue
                seen_ids.add(key)

                try:
                    score = float(fields[5])  # full-seq score
                    bias  = float(fields[6])  # full-seq bias
                except ValueError:
                    continue

                if bias <= score * bias_ratio:
                    outfile.write(line)

    os.replace(temp_file, tblout_file)

def filter_by_bitscore(tblout_file, min_bitscore=50):
    """Filter HMMER tblout output by bitscore."""
    temp_file = f"{tblout_file}.filtered"
    with open(tblout_file, "r") as infile, open(temp_file, "w") as outfile:
        for line in infile:
            if line.startswith("#"):
                outfile.write(line)  # Keep header lines
                continue
            
            fields = line.split()
            if len(fields) >= 6:  # Ensure enough fields for bitscore
                try:
                    bitscore = float(fields[5])
                    if bitscore >= min_bitscore:
                        outfile.write(line)
                except ValueError:
                    continue

    os.replace(temp_file, tblout_file)

def run_hmmer(proteins, polymer, outdir, evalue, cores, bitscore):
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hmm_dir = os.path.join(package_dir, "data", "polymer_hmms")
    
    # Assume polymer is a single polymer (e.g., "PET")
    poly = polymer.strip().upper()
    hmm_files = []
    polymer_to_hmm = {
         "P3HP": ["P3HP.hmm"], "PBAT": ["PBAT.hmm"],
         "PBS": ["PBS.hmm"], "PBSA": ["PBSA.hmm"], "PCL": ["PCL.hmm"],
         "PEA": ["PEA.hmm"], "PET": ["PET.hmm"], "PHA": ["PHA.hmm"],
         "PHB": ["PHB.hmm"], "PLA": ["PLA.hmm"],
         "PHBV": ["PHBV.hmm"]
    }
    if poly in polymer_to_hmm:
         for hmm in polymer_to_hmm[poly]:
             hmm_path = os.path.join(hmm_dir, hmm)
             if os.path.exists(hmm_path):
                  hmm_files.append(hmm_path)
    else:
         print(f"Warning: Polymer {poly} not supported for HMMER search.")
    
    os.makedirs(outdir, exist_ok=True)
    all_output_files = []
    
    # Use the current polymer value for output file names.
    for hmm_file in hmm_files:
        hmm_base = os.path.splitext(os.path.basename(hmm_file))[0]
        tblout_file = os.path.join(outdir, f"{poly}_HMMER.tblout")
        log_file = os.path.join(outdir, "hmmer.log")
        cmd = [
            "hmmsearch",
            "-E", str(evalue),
            "--incE", str(evalue),
            "--cpu", str(cores),
            "--noali",
            "--tblout", tblout_file,
            hmm_file,
            proteins
        ]
        
        # Remove log file if it exists
        if os.path.exists(log_file):
            os.remove(log_file)
        
        try:
            with open(log_file, "a") as log:
                log.write(f"🧪Running command: {' '.join(cmd)}\n")
                log.write(f"🧪Using bitscore filter: {bitscore}\n")
                subprocess.check_call(cmd, stdout=log, stderr=log)
        except subprocess.CalledProcessError as e:
            with open(log_file, "a") as log:
                log.write(f"❌ERROR: HMMER command failed: {e}\n")
        
        # If the output file wasn't created, generate an empty one.
        if not os.path.exists(tblout_file):
            print(f"Warning: No HMMER output for polymer {poly}; creating empty file {tblout_file}.")
            with open(tblout_file, "w") as f:
                f.write("# HMMER tblout output - No hits found\n")
        else:
            filter_hmmer_output_inplace(tblout_file)
            filter_by_bitscore(tblout_file, min_bitscore=bitscore)
        
        all_output_files.append(tblout_file)
    
    print(f"✅HMMER completed successfully for polymer: {poly}.")
    return all_output_files





<p align="center">
  <img src="https://github.com/user-attachments/assets/bb5be2a1-3783-457e-85d8-e2278691697a" alt="logo-transparent" width="300">
</p>

## What is PlasticEnz?
PlasticEnz offers a streamlined and accessible solution for identifying plastic-degrading enzymes in metagenomic data by combining homology-based and machine learning approaches. 
It accepts contigs, genomes, MAGs and proteins and screens them for potential plastic degrading homologous enzymes.

![Figure 1](https://github.com/user-attachments/assets/3291f071-7194-463b-93b8-aab7e2f03c3f)


## Downloading PlasticEnz

Clone the repositiory and navigate into the tool main folder (where setup.py is located)
```bash
git clone https://github.com/akrzyno/PlasticEnz.git
```
Set Up the Conda Environment with External Tools
 ```bash
conda create -n plasticenz_env --no-channel-priority -c bioconda -c conda-forge -c defaults python=3.11 libffi=3.4.2 prodigal hmmer diamond bowtie2 samtools
```
Activate the environment
```bash
conda activate plasticenz_env
```
Install Python Package Dependencies
With your conda environment activated, navigate to the package folder and install the remaining python packages:
```bash
cd PlasticEnz
pip install -r requirements.txt
```
Install the package
```bash
pip install .
```
Test if it runs correctly

  To see all the options:
 ```bash
  plasticenz
 ```
or
 ```bash
  plasticenz --help
 ```

## Running a test-case
Please before using the PlasticEnz on your dataset run the test-case (data included within the package) to ensure all is sound. To do so run:

```bash
  plasticenz --test --outdir .
```
Wait until you see "✅PlasticEnz analysis completed successfully!" and check the outdir folder for the output folder. 
If you see three these files there: 
Abundances_table.tsv	
Proteins_unique.fa	
Summary_table.tsv,you are good to go.
## All options
```

       ___ _           _   _        __          
      / _ | | __ _ ___| |_(_) ___  /___ __  ____
     / /_)| |/ _` / __| __| |/ __|/_\| '_ \|_  /
    / ___/| | (_| \__ | |_| | (__//__| | | |/ / 
    \/    |_|\__,_|___/\__|_|\___\__/|_| |_/___|
                                                
    
    #####################################
    #        Welcome to PlasticEnz      #
    #####################################
    
    

Please remember to cite following tools:
- Prodigal: Hyatt et al., 2010. BMC Bioinformatics. DOI: 10.1186/1471-2105-11-119
- HMMER: Eddy, 2011. PLoS Comput Biol. DOI: 10.1371/journal.pcbi.1002195
- DIAMOND: Buchfink et al., 2015. Nat Methods. DOI: 10.1038/nmeth.3176
- Bowtie2: Langmead & Salzberg, 2012. Nat Methods. DOI: 10.1038/nmeth.1923
- Samtools: Danecek et al., 2021. Gigascience. DOI: 10.1093/gigascience/giab008
- ProtTrans: Elnaggar et al., 2022. IEEE TPAMI. DOI: 10.1109/TPAMI.2021.3095381


usage: plasticenz [-h] [-c CONTIGS] [-1 READS_FORWARD] [-2 READS_REVERSE] [-p PROTEINS] [-g GENOME] [--cores CORES] [--polymer POLYMER] [--outdir OUTDIR]
                  [--use_gpu] [--evalue_hmmer EVALUE_HMMER] [--bitscore_hmmer BITSCORE_HMMER] [--evalue_diamond EVALUE_DIAMOND]
                  [--bitscore_diamond BITSCORE_DIAMOND] [--test] [--sensitive]

PlasticEnz: A tool for detecting plastic-degrading enzymes from sequence data.

options:
  -h, --help            show this help message and exit
  -c CONTIGS, --contigs CONTIGS
                        Path to contigs file (FASTA). (default: None)
  -1 READS_FORWARD, --reads_forward READS_FORWARD
                        Path to forward reads file (FASTQ). (default: None)
  -2 READS_REVERSE, --reads_reverse READS_REVERSE
                        Path to reverse reads file (FASTQ). (default: None)
  -p PROTEINS, --proteins PROTEINS
                        Path to protein file (FASTA). (default: None)
  -g GENOME, --genome GENOME
                        Path to genome or MAG file (FASTA). (default: None)
  --cores CORES         Number of CPU cores to use. (default: 1)
  --polymer POLYMER     Polymer(s) to screen for. Use 'all' for all available. (default: None)
  --outdir OUTDIR       Output directory. (default: None)
  --use_gpu             Attempt to use GPU for accelerated computations. (default: False)
  --evalue_hmmer EVALUE_HMMER
                        E-value threshold for HMMER search. (default: 1e-05)
  --bitscore_hmmer BITSCORE_HMMER
                        Bitscore value for HMMER search. (default: 20)
  --evalue_diamond EVALUE_DIAMOND
                        E-value threshold for DIAMOND search. (default: 1e-05)
  --bitscore_diamond BITSCORE_DIAMOND
                        Minimum alignment quality for DIAMOND search. (default: 20)
  --test                Run the tool with a predefined test dataset. (default: False)
  --sensitive           Use neural network model (nn_model.pkl) for sensitive predictions. (default: False)
```
## ❓Troubleshooting

PlasticEnz requires several external tools. If you encounter issues with Conda installation, you can install them manually:
```bash
conda install -c bioconda prodigal=2.6.3
conda install -c bioconda hmmer=3.4
conda install -c bioconda diamond=2.1.8
conda install -c bioconda bowtie2=2.5.4
conda install -c bioconda samtools=1.21
 ```




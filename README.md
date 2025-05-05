![logo-transparent](https://github.com/user-attachments/assets/bb5be2a1-3783-457e-85d8-e2278691697a)

# How to Download and Run PlasticEnz

## Option A: Using Conda/pip downloads

1. Clone the repositiory and navigate into the tool main folder (where setup.py is located)
```bash
git clone https://github.com/akrzyno/PlasticEnz.git
```
3. Set Up the Conda Environment with External Tools
 ```bash
conda create -n plasticenz_env --no-channel-priority -c bioconda -c conda-forge -c defaults python=3.11 libffi=3.4.2 prodigal hmmer diamond bowtie2 samtools
```
4. Activate the environment
```bash
conda activate plasticenz_env
```
5. Install Python Package Dependencies
With your conda environment activated, navigate to the package folder and install the remaining python packages:
```bash
cd PlasticEnz
pip install -r requirements.txt
```
6. Install the package
```bash
pip install .
```
7. Test if it runs correctly

  To see all the options:
  ```bash
  plasticenz OR plasticenz --help
  ```
  To run a test-case:
  ```bash
  plasticenz --test --outdir .
  ```

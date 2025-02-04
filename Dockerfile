# 1️⃣ Use an official lightweight Linux image with Conda
FROM continuumio/miniconda3

# 2️⃣ Set environment variables to avoid prompts during Conda installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# 3️⃣ Install system dependencies
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4️⃣ Create a Conda environment with HMMER, Prodigal, DIAMOND, and Python 3.11.11
RUN conda create -n plasticenz_env -c bioconda -c conda-forge -c defaults \
    python=3.11.11 prodigal hmmer diamond && conda clean --all -y

# 5️⃣ Activate Conda environment and install Python dependencies
SHELL ["conda", "run", "-n", "plasticenz_env", "/bin/bash", "-c"]

# 6️⃣ Copy and install Python dependencies (using caching)
COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --default-timeout=1000 -r /tmp/requirements.txt

# 7️⃣ Clone PlasticEnz from GitHub
RUN git clone https://github.com/AMK06-1993/PlasticEnz.git /PlasticEnz

# 8️⃣ Set working directory & install PlasticEnz
WORKDIR /PlasticEnz
RUN pip install .

# 9️⃣ Ensure correct permissions
RUN chmod -R 777 /PlasticEnz

# 🔟 Define entry point for the container
ENTRYPOINT ["conda", "run", "-n", "plasticenz_env", "plasticenz"]

import os
import pandas as pd
import pysam
from Bio import SeqIO

def parse_prodigal_gene_fna(fna_path):
    genes = []
    for record in SeqIO.parse(fna_path, "fasta"):
        header = record.description
        parts = header.split(" # ")
        contig = "_".join(parts[0].split("_")[:-1])
        start = int(parts[1])
        end = int(parts[2])
        strand = int(parts[3])
        gene_id = record.id
        genes.append({
            "contig": contig,
            "start": min(start, end),
            "end": max(start, end),
            "strand": strand,
            "gene_id": gene_id,
            "length": abs(end - start + 1)
        })
    return genes

def count_reads_per_gene(bam_file, genes):
    bam = pysam.AlignmentFile(bam_file, "rb")
    counts = {}
    for gene in genes:
        count = bam.count(
            reference=gene["contig"],
            start=gene["start"] - 1,
            end=gene["end"]
        )
        if count > 0:
            counts[gene["gene_id"]] = {
                "raw_count": count,
                "length": gene["length"],
                "contig": gene["contig"],
                "start": gene["start"],
                "end": gene["end"],
                "strand": gene["strand"]
            }
    bam.close()
    return counts

def compute_normalized_abundances(gene_counts):
    total_reads = sum(v["raw_count"] for v in gene_counts.values())
    rpk_sum = sum(v["raw_count"] / (v["length"] / 1000) for v in gene_counts.values())

    rows = []
    for gene_id, v in gene_counts.items():
        raw = v["raw_count"]
        length_kb = v["length"] / 1000
        rpk = raw / length_kb
        rpkm = rpk / (total_reads / 1e6) if total_reads > 0 else 0
        tpm = (rpk / rpk_sum * 1e6) if rpk_sum > 0 else 0
        cpm = (raw / total_reads * 1e6) if total_reads > 0 else 0

        row = {
            "Protein Name": gene_id,
            "contig": v["contig"],
            "start": v["start"],
            "end": v["end"],
            "strand": v["strand"],
            "raw_count": raw,
            "cpm": cpm,
            "rpk": rpk,
            "rpkm": rpkm,
            "tpm": tpm
        }
        rows.append(row)
    return pd.DataFrame(rows)

def run_abundance_from_bam(bam_files, gene_fna, outdir, cores=1):
    os.makedirs(outdir, exist_ok=True)
    genes = parse_prodigal_gene_fna(gene_fna)

    all_dfs = []
    for bam in bam_files:
        sample = os.path.basename(bam).split(".")[0]
        print(f"🔄 Processing: {sample}")
        gene_counts = count_reads_per_gene(bam, genes)
        df = compute_normalized_abundances(gene_counts)
        df["sample"] = sample
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    output_path = os.path.join(outdir, "abundance_from_bam.tsv")
    combined_df.to_csv(output_path, sep="\t", index=False)
    return output_path



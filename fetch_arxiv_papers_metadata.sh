#!/bin/bash -l
#SBATCH --job-name=arxiv_fetch_papers
#SBATCH --output=arxiv_fetch_papers.log
#SBATCH --partition=glacier
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

#module load python/3.10

# Full path to your extractor script:
SCRIPT=/home/tom_menczel/big_data_project/ArXivAnalysis/src/extract_with_abstract.py

# Ensure it's executable
chmod +x "$SCRIPT"

# Run the fetch_papers subcommand
#python "$SCRIPT" fetch_papers \
#    --data-dir /mnt/c/Datasets/big_data_project_52017/2024_25/arxiv_data \
#    --from-date 2021-01-01 \
#    --until-date 2021-12-31 \
#    --category cs.LG \
#    --id-regex '^2104\.\d+' \
#    --what metadata \
#    --max-results 100 \
#    --batch-size 50

# Run the script with the correct argument names
python3 "$SCRIPT" \
    --id '^2104\.\d+' \
    --start 2021-01-01 \
    --end 2021-12-31 \
    --category cs.LG \
    --formats metadata \
    --output-dir /mnt/c/Datasets/big_data_project_52017/2024_25/arxiv_data \
    --verbose

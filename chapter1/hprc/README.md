# chapter1/hprc/

Slurm array scripts for the paper's final (`v2`) runs on TAMU HPRC (FASTER and Grace, one A100 per task). Each file
documents its array-index mapping and its outputs at the top. Every script sources `hprc/activate.sh`; set up the
cluster environment once with `hprc/hprc.env.example` and `hprc/setup_env.sh`, and replace `YOUR_ACCOUNT` in the
`#SBATCH --account` lines. The submission commands for each paper element are in the top-level `README.md`
("Retraining").

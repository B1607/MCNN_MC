python exfold.py esm1b_t33_650M_UR50S /mnt/D/jupyter/Melanoma/fasta/Train/neg /mnt/D/jupyter/Melanoma/data/esm1b/train/neg --repr_layers 33 --include per_tok

python exfold.py esm1b_t33_650M_UR50S /mnt/D/jupyter/Melanoma/fasta/Test/neg /mnt/D/jupyter/Melanoma/data/esm1b/test/neg --repr_layers 33 --include per_tok

python exfold.py esm1b_t33_650M_UR50S /mnt/D/jupyter/Melanoma/fasta/Train/pos /mnt/D/jupyter/Melanoma/data/esm1b/train/pos --repr_layers 33 --include per_tok

python exfold.py esm1b_t33_650M_UR50S /mnt/D/jupyter/Melanoma/fasta/Test/pos /mnt/D/jupyter/Melanoma/data/esm1b/test/pos --repr_layers 33 --include per_tok
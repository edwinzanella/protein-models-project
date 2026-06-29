# Active Site Graph Neural Network

Predicts, per residue, whether a protein is part of an active/binding site using a GNN trained on 3D structures from the RCSB Protein Data Bank.

## Docker Structure

A Docker container is set up creating a docker image of an OS running Python 3.12. The Dockerfile install all of the python dependencies listed in requirements.txt.

There was a deadlock error running the files on MacOS which required me to switch my file shared implementation on Docker from VirtioFS to gRPC FUSE. If you receive a deadlock error, try switching between these two file sharing implementations.

## Reposity

The repository for this codebase is at https://github.com/edwinzanella/protein-models-project and uses GitHub Large File Storage on .txt files. 

## Shell

run.sh needs execution permissions so run:
chmod +x run.sh

## Data folder

Download the BioLiP.txt file downloaded from https://zhanggroup.org/BioLiP/. Enter it into the /data folder where the BioLiPREADME.txt and pdb_ids.txt files are located. Columns 0, 1, and 7 are used from the BioLiP.txt file.

Column 0: PDB ID
Column 1: Receptor Chain
Column 7: Binding Site Residues

pdb_ids.txt uses this format per line:
`pdb_id,chain_id`

from columns 0 and 1 of BioLiP.txt which are manually entered into into pdb_ids.txt for training.

So far I have manually added the first 250 entries into pdb_ids.txt

## Running Everything

Download structures:
./run.sh python src/download_structures.py --ids_file data/pdb_ids.txt --out_dir data/pdb_raw

Build the graph dataset:
./run.sh python src/build_dataset.py \
    --ids_file data/pdb_ids.txt \
    --pdb_dir data/pdb_raw \
    --biolip_file data/BioLiP.txt \
    --out_file data/dataset.pt

Train:
./run.sh python src/train.py --dataset_file data/dataset.pt --epochs 50
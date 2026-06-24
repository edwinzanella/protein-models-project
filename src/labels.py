import argparse
import pandas as pd

COL_PDB_ID = 0
COL_CHAIN = 1
COL_BINDING_RESIDUES = 7

class ActiveSiteLabels:
    def __init__(self, biolip_file):
        self.df = pd.read_csv(biolip_file, sep="\t", header=None, dtype=str)
        
    def get_active_site_residues(self, pdb_id, chain_id):
        pdb_id = pdb_id.lower()
        matches = self.df[
            (self.df[COL_PDB_ID].str.lower() == pdb_id)
            & (self.df[COL_CHAIN] == chain_id)
        ]
        
        residue_numbers = set()
        for raw in matches[COL_BINDING_RESIDUES].dropna():
            for token in raw.split():
                digits = "".join(c for c in token if c.isdigit())
                if digits:
                    residue_numbers.add(int(digits))
        
        return residue_numbers
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick lookup test for active site labels")
    parser.add_argument("--biolip_file", required=True)
    parser.add_argument("--pdb_id", required=True)
    parser.add_argument("--chain_id", required=True)
    args = parser.parse_args()
    
    labels = ActiveSiteLabels(args.biolip_file)
    residues = labels.get_active_site_residues(args.pdb_id, args.chain_id)
    print(f"Active site residues for {args.pdb_id} chain {args.chain_id}:{sorted(residues)}")
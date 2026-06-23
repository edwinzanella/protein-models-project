import argparse
import os
import torch

from labels import ActiveSiteLabels
from graph_builder import structure_to_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", required=True)
    parser.add_argument("--pdb_dir", default="data/pdb_raw")
    parser.add_argument("--biolip_file", required=True)
    parser.add_argument("--out_file", default="data/dataset.pt")
    parser.add_argument("--contact_threshold", type=float, default=8.0)
    args = parser.parse_args()
    
    label_lookup = ActiveSiteLabels(args.biolip_file)
    
    graphs = []
    skipped = []
    
    with open(args.ids_file) as f:
        entries = [line.strip().split(",") for line in f if line.strip()]
        
    print(f"Building graphs for {len(entries)} structures...")
    
    for i, (pdb_id, chain_id) in enumerate(entries):
        cif_path = os.path.join(args.pdb_dir, f"{pdb_id}.cif")
        if not os.path.exists(cif_path):
            skipped.append((pdb_id, "missing .cif file"))
            continue
    
        actives_residues = label_lookup.get_active_site_residues(pdb_id, chain_id)
        if not actives_residues:
            skipped.append((pdb_id, "no active site labels found"))
            continue
        
        graph = structure_to_graph(
            cif_path, chain_id, actives_residues, contact_threshold=args.contact_threshold
        )
        if graph is None:
            skipped.append((pdb_id, "graph build failed (too few residues / bad chain)"))
            continue
        
        graphs.append(graph)
        print(f"[{i + 1}/{len(entries)}] {pdb_id}:{chain_id} -> {graph.num_nodes} residues, "
              f"{int(graph.y.sum())} active site residues")
        
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    torch.save(graphs, args.out_file)
    
    print(f"\nSaved {len(graphs)} graphs to {args.out_file}")
    if skipped:
        print(f"Skipped {len(skipped)} entries:")
        for pdb_id, reason in skipped:
            print(f"{pdb_id}:{reason}")
    
if __name__ == "__main__":
    main()
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import three_to_one

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}

def _aa_one_hot(resname):
    one_hot = np.zeros(len(AA_LIST), dtype=np.float32)
    try:
        one_letter = three_to_one(resname)
        idx = AA_TO_INDEX.get(one_letter)
        if idx is not None:
            one_hot[idx] = 1.0
    except KeyError:
        pass
    return one_hot

def structure_to_graph(cif_path, chain_id, active_site_residues, contact_threshold=8.0):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_path)
    
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return None
    
    coords = []
    features = []
    labels = []
    res_numbers = []
    
    for res in chain:
        if res.id[0] != " ":
            continue
        if "CA" not in res:
            continue
        
        coords.append(res["CA"].coord)
        features.append(_aa_one_hot(res.get_resname()))
        res_number = res.id[1]
        res_numbers.append(res_number)
        labels.append(1 if res_number in active_site_residues else 0)
        
    if len(coords) < 10:
        return None
    
    coords = np.array(coords, dtype=np.float32)
    x = torch.tensor(np.array(features), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    pos = torch.tensor(coords, dtype=torch.float)
    
    edge_index = _build_edges(coords, contact_threshold)
    
    return Data(x=x, edge_index=edge_index, y=y, pos=pos)

def _build_edges(coords, threshold):
    from scipy.spatial import cKDTree
    
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=threshold, output_type="ndarray")
    
    if len(pairs) == 0:
        edge_list = [[i, i] for i in range (len(coords))]
    else:
        edge_list = np.concatenate([pairs, pairs[:, ::-1]], axis=0).tolist()
        
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
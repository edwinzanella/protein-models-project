import argparse
import os
import time
import requests

def download_pdb(pdb_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pdb_id}.cif")
    if os.path.exists(path):
        return path
    
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(path, "wb") as f:
        f.write(response.content)
        
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_file", required=True, help="Text file with one PDB ID per line")
    parser.add_argument("--out_dir", default="data/pdb_raw")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to wait between response")
    args = parser.parse_args()
    
    with open(args.ids_file) as f:
        content = f.read()
    
    pdb_ids = [line.strip() for line in content.splitlines() if line.strip()]
        
    print(f"Downloading {len(pdb_ids)} structures...")
    failed = []
    for i, pdb_id in enumerate(pdb_ids):
        try:
            download_pdb(pdb_id, args.out_dir)
            print(f"[{i + 1}/{len(pdb_ids)}] {pdb_id} OK")
        except Exception as e:
            print(f"[{i + 1}/{len(pdb_ids)}] {pdb_id} FAILED as {e}")
            failed.append(pdb_id)
        time.sleep(args.delay)
    
    if failed:
        print(f"\n{len(failed)} downloads failed: {failed}")

if __name__ == "__main__":
    main()
import os
import json


def patch_missing_config(run_dir):
    run_files = os.listdir(run_dir)
    if "config.json" in run_files:
        return
    print(f"Missing config.json in run: {run_dir}")
    # Find the config.json file in genomes directory
    if not "genomes" in run_files:
        raise Exception(f"Missing genomes directory in run: {run_dir}")
    genomes = os.listdir(os.path.join(run_dir, "genomes"))
    if len(genomes) == 0:
        raise Exception(f"Empty genomes directory in run: {run_dir}")

    # Find the first genome file
    genome_file = genomes[0]
    with open(os.path.join(run_dir, "genomes", genome_file)) as f:
        genome = f.read()
    genome = json.loads(genome)
    config = genome["config"]
    # save to run dir
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f)



def submit_run_resume(run_path):
    patch_missing_config(run_path)
    submit_cmd = f"sbatch scripts/resume-move-experiment.sh {run_path}"
    print(submit_cmd)
    # os.system(submit_cmd)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--delete", "-d", action="store_true", default=False, help="Delete existing empty runs")
    
    args = parser.parse_args()
    
    conds_dir = os.path.join(args.out_dir, "conditions")

    for out_cond_dir in os.listdir(conds_dir):
        out_cond_dir = os.path.join(conds_dir, out_cond_dir)
        if not os.path.isdir(out_cond_dir):
            continue
        print(out_cond_dir)
        for run in os.listdir(out_cond_dir):
            if not os.path.isdir(os.path.join(out_cond_dir, run)):
                continue
            run_files = os.listdir(os.path.join(out_cond_dir, run))
            if len(run_files) == 0:
                if args.delete:
                    # os.rmdir(os.path.join(out_cond_dir, run))
                    print("Deleted empty run: ", os.path.join(out_cond_dir, run))
                else:
                    print(f"Empty run: {run}")
                continue
            if not 'fitness_by_batch.pt' in run_files:
                print(f"Missing fitness_by_batch.pt in run: {run}")
                submit_run_resume(os.path.join(out_cond_dir, run))
            else:
                print(f"Already finished run: {run}")

    
    
    
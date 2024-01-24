import os




def submit_run(filename, out_dir):
    submit_cmd = f"sbatch scripts/submit-move-experiment.sh {filename} {out_dir}"
    print(submit_cmd)
    # os.system(submit_cmd)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("runs", type=int)
    parser.add_argument("--delete", "-d", action="store_true", default=False, help="Delete existing empty runs")
    
    args = parser.parse_args()
    
    conds_dir = os.path.join(args.out_dir, "conditions")
    if not os.path.exists(conds_dir):
        os.makedirs(conds_dir)
    
    done = {}
    
    # build dict of done runs by target and condition
    for tar_cond in os.listdir(args.in_dir):
        tar_cond = os.path.join(args.in_dir, tar_cond)
        if os.path.isdir(tar_cond):
            continue
        
        tar = tar_cond.split("_")[0].split("/")[-1]
        cond = "_".join(tar_cond.split("_")[1:]).replace(".json", "")
        done[(tar,cond)] = 0

    for out_cond_dir in os.listdir(conds_dir):
        out_cond_dir = os.path.join(conds_dir, out_cond_dir)
        if not os.path.isdir(out_cond_dir):
            continue
        print(out_cond_dir)
        for run in os.listdir(out_cond_dir):
            if not os.path.isdir(os.path.join(out_cond_dir, run)):
                continue
            run_files = os.listdir(os.path.join(out_cond_dir, run))
            if len(run_files) == 0 or "target.txt" not in run_files:
                if args.delete:
                    # os.rmdir(os.path.join(out_cond_dir, run))
                    print("Deleted empty run: ", os.path.join(out_cond_dir, run))
                else:
                    print(f"Empty run: {run}")
                continue
            target_file = [f for f in run_files if "target.txt" in f][0]
            
            target = ""
            with open(os.path.join(out_cond_dir, run, target_file)) as f:
                target = f.read().strip().split("/")[-1].split(".")[0]
            print(target)
            
            done[(target, out_cond_dir.split("/")[-1])] += 1


    print("Done runs:")
    for k, v in done.items():
        print(f"{k[0]}\t{k[1]}:\t{v}")
        
    
    
    
    
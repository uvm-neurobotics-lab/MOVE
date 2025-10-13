import os


target_exception_names = [
    "linear_sin" 
] # truly terrible that we have to do this


def submit_run(filename, out_dir, dry=True):
    submit_cmd = f"sbatch scripts/submit-move-experiment.sh {filename} {out_dir}"
    print("\t>\t\t", submit_cmd)
    if not dry:
        os.system(submit_cmd)
    

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
    path_map = {} # (target, condition) -> filename
    
    # build dict of done runs by target and condition
    for tar_cond in os.listdir(args.in_dir):
        tar_cond = os.path.join(args.in_dir, tar_cond)
        if os.path.isdir(tar_cond):
            continue
        
        config_file_name = tar_cond.split("/")[-1].replace(".json", "").strip()

        t_exception = None
        for t in target_exception_names:
            if t in config_file_name:
                t_exception = t
                break
        
        tar, cond = None, None
        if t_exception is not None:
            tar = t_exception
            cond = config_file_name.replace(t_exception+"_","")
        else:
            tar = config_file_name.split("_")[0] # first part is target
            cond = "_".join(config_file_name.split("_")[1:]) # rest is condition

        done[(tar,cond)] = 0
        path_map[(tar,cond)] = tar_cond

    for out_cond_dir in os.listdir(conds_dir):
        out_cond_dir = os.path.join(conds_dir, out_cond_dir)
        if not os.path.isdir(out_cond_dir):
            continue
        print("out condition dir:", out_cond_dir)
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
            condition_name = out_cond_dir.split("/")[-1]
            done[(target, condition_name)] += 1


    print("Done runs:")
    sorted_done_keys = sorted(done.keys(), key=lambda x: (x[0], x[1]))
    last_k_t = None
    total_missing = 0

    missing = {}
    for k in sorted_done_keys:
        v = done[k]
        p = f"\t\t{k[0]:<15} {k[1]:<30} {v}/{args.runs}"
        if last_k_t != k[0]:
            print(f"\nTarget: {k[0]}")
            last_k_t = k[0]
        # print(f"{k[0]}\t{k[1]}:\t{v}")
        # better formatting:
        print(p)
        total_missing += max(0, args.runs - v)
        if v < args.runs:
            missing[path_map[k]] = args.runs - v
    print(f"{'-'*80}\nTotal missing runs: {total_missing}\n")


    confirm = input("Submit missing runs? (y/n): ")

    dry = True
    if confirm.lower().strip().startswith("y"):
        dry = False
    

    for pathname, count in missing.items():
        count = missing[pathname]
        if not dry:
            print(f"Submitting {count} runs for {pathname}")
        else:
            print(f"(Dry run) Would submit {count} runs for {pathname}")
        for _ in range(count):
            submit_run(pathname, args.out_dir, dry=dry)
        print()
    
    
    
    
import json
import os
import itertools
import argparse
import shutil

DEFAULT_TEMPLATE = "experiments/template-move.json"

DEFAULT_TARGET = "data/skull.png"





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=str, help='Path to plan json file.')
    parser.add_argument('--template', '-t', type=str, default=DEFAULT_TEMPLATE, help='Path to template json file (default: default.json).')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to output json file (default: ../results/<experiment-name>).')
    parser.add_argument('--generated-output', '-g', type=str, default=None, help='Path to output json file (default: experiments/generated/<experiment-name>).')
    
    
    args = parser.parse_args()
    
    assert os.path.exists(args.plan), f"Plan file {args.plan} does not exist"
    assert os.path.exists(args.template), f"Template file {args.template} does not exist"

    plan = None
    with open(args.plan) as f:
        plan = json.load(f)
    
    
    addl_controls = plan["additional_controls"]
    addl_params = plan.get("additional_parameters", {})
    conditions = []
    name_template = plan.get("name_template", "{name}")

    if args.output == None:
        args.output = os.path.join("..", "results", plan["experiment_name"])
    
    if args.generated_output == None:
        args.generated_output = os.path.join("experiments",'generated', plan["experiment_name"])
    
    
    os.makedirs(args.output, exist_ok=True)
    
    serial_output = os.path.join(args.generated_output, "serial")
    parallel_output = os.path.join(args.generated_output)
    
    # empty the dirs
    shutil.rmtree(parallel_output, ignore_errors=True)
    shutil.rmtree(serial_output, ignore_errors=True)
    
    # create empty dirs
    os.makedirs(parallel_output)
    os.makedirs(serial_output)
    

    if plan["mode"] == 'grid':
        skip_levels = plan.get("skip_levels", [])
        # cartesian product of all variables/levels
        level_pairs = list(itertools.product(*plan["levels"]))
        for pair in level_pairs:
            if pair in skip_levels:
                continue
            name = name_template.format(**dict(zip(plan["variables"], pair)))
            vars = dict(zip(plan["variables"], pair))
            for variable in addl_params:
                for param in addl_params[variable]:
                    vars[param] = addl_params[variable][param][str(vars[variable])]
           
            cond_dict = {name:vars}
            conditions.append(cond_dict)
            
    if plan["mode"] == "enum":
        # list of conditions
        for cond in plan["conditions"]:
            cond_dict = cond
            conditions.append(cond_dict)
    
    template_string = None
    with open(args.template) as f:
        template_string = f.read()
    
    
    for tar, cm in zip( plan["targets"], plan["color_modes"]):
        assert os.path.exists(tar), f"Target file {tar} does not exist"
            
        save_string = template_string
        
        save_string = save_string.replace("<EXPERIMENT_NAME>", plan["experiment_name"])
        save_string = save_string.replace("<TARGET>", tar)
        save_string = save_string.replace("<COLOR_MODE>", cm)
        save_string = save_string.replace("<OUTPUT>", args.output)
        
        if len(addl_controls) > 0:
            addl_controls_string = ""
            addl_controls_string = json.dumps(addl_controls, indent=4)[1:-1]
            # for control in addl_controls:
                # addl_controls_string += f"\n\t\"{control}\": {addl_controls[control]}" + ("," if control != list(addl_controls.keys())[-1] else "")
            save_string = save_string.replace("<ADDITIONAL_CONTROLS>", addl_controls_string)
        else:
            save_string = save_string.replace(", <ADDITIONAL_CONTROLS>", "")
            
        parallel_save_string = save_string
        save_string = save_string.replace("<CONDITIONS>", json.dumps(conditions, indent=4))
        
        tar_pretty = tar.split('/')[-1].split('.')[0]
        save_path = os.path.join(serial_output, f"{tar_pretty}.json")
        print(f"Saving to {save_path}")
        with open(save_path, 'w') as f:
            f.write(save_string)
        
        for condition in conditions:
            this_parallel_save_string = parallel_save_string
            this_parallel_save_string = this_parallel_save_string.replace("<CONDITIONS>", json.dumps([condition], indent=4))

            with open(os.path.join(parallel_output, f"{tar_pretty}_{list(condition.keys())[0]}.json"), 'w') as f:
                f.write(this_parallel_save_string)

        
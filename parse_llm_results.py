from helpers.llm_helper import parse_llm_output
import os
import json
from ruamel.yaml import YAML
import argparse

# Set llm_output filename
def main(llm_outputs_filename):
    base_path = os.getcwd()
    llm_outputs_filename = llm_outputs_filename
    parsed_outputs_filename = f"parsed_{llm_outputs_filename.strip('.json')}.yaml"
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Load Data
    print("Loading Data...")
    llm_outputs_path = os.path.join(base_path, "llm_results", llm_outputs_filename)
    with open(llm_outputs_path, "r") as f:
        llm_outputs = json.load(f)
    print("Done")

    # Iterate through the outputs and extract yaml format
    # Sort the json file by mrn in ascending order
    llm_outputs = dict(sorted(llm_outputs.items(), key=lambda x: int(x[0].split("acc")[1])))

    all_parsed_output = {}
    for mrn, output in llm_outputs.items():
        try:
            parsed_output = parse_llm_output(output)
            if parsed_output is None:
                print(f"{mrn} is None")
        except Exception as e:
            print(mrn)
            print(e)
        all_parsed_output[mrn] = parsed_output

    with open(os.path.join(base_path, "llm_results", parsed_outputs_filename), "w") as f:
        yaml.dump(all_parsed_output, f)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Set a filename of the llm_output to parse")
    parser.add_argument("--filename", type=str, required=True,
                        help="Set filename of the llm output to parse (relative path within llm_results directory)")
    args = parser.parse_args()
    main(str(args.filename))
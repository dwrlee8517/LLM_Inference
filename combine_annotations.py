from ruamel.yaml import YAML
import argparse
import os
import glob

def main(dev=False):
    yaml = YAML()
    combined_yaml = {}
    if not dev:
        print("Concatenating test set")
        save_filename = "/raid/dongwoolee/RadPath/manual_annotations/annotations_test_combined.yaml"
        all_files = glob.glob("/raid/dongwoolee/RadPath/manual_annotations/annotations_test_*.yaml")
        print(all_files)
    else:
        print("Concatenating dev set")
        save_filename = "/raid/dongwoolee/RadPath/manual_annotations/annotations_dev_combined.yaml"
        all_files = glob.glob("/raid/dongwoolee/RadPath/manual_annotations/annotations_dev_*.yaml")
        print(all_files)

    
    all_files = [f for f in all_files if f != save_filename]
    for file in all_files:
        with open(file, "r") as f:
            data = yaml.load(f)
        combined_yaml = {**combined_yaml, **data}
    
    sorted_combined_yaml = dict(sorted(combined_yaml.items(), key=lambda x: int(x[0].split("acc")[1])))
    with open(save_filename, "w") as f:
        yaml.dump(sorted_combined_yaml, f)
    print(f"File saved to {save_filename}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", default=False)
    args = parser.parse_args()
    main(args.dev)
import argparse
import numpy as np
from ruamel.yaml import YAML
from helpers.annotation_metrics import *
from helpers.annotation_dataset import *
from helpers.llm_helper import *
import os
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import glob

base_path = os.getcwd()
llm_results_dir = os.path.join(base_path, "llm_results")
manual_annotation_dir = os.path.join(base_path, "manual_annotations")

def main(manual_filename, llm_filename):

    print("Running main with the following files:")
    print("Manual annotation file:", manual_filename)
    print("LLM annotation file:", llm_filename)
    
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    
    print("Loading Dataset...")
    with open(os.path.join(manual_annotation_dir, manual_filename), "r") as f:
        manual_annotations = yaml.load(f)
    with open(os.path.join(llm_results_dir, llm_filename), "r") as f:
        llm_annotations = yaml.load(f)
    print("Done")

    print("Loading to RadPathDataset object")
    print("Manual Dataset Errors:")
    dataset_manual = RadPathDataset(manual_annotations)
    print("LLM Dataset Errors:")
    dataset_llm = RadPathDataset(llm_annotations)

    # Get MRN of the patients in the manual annotation set
    mrns = sorted(list(dataset_llm.mrns), key=lambda x: int(x.split("acc")[1]))

    # Initialize Metrics Obejcts
    rad_loc_acc = LocationAccuracy()
    rad_size_acc = BaseAccuracy()
    comp_acc = CompAccuracy()
    echo_acc = EchoAccuracy()
    shape_acc = ShapeAccuracy()
    margin_acc = MarginAccuracy()
    foci_acc = FociAccuracy()
    tirads_acc = TIRADSAccucarcy()
    match_metrics = MatchAccuracy()
    
    bx_loc_acc = LocationAccuracy()
    bx_size_acc = BaseAccuracy()
    bx_result_acc = BxResultAccuracy()

    ln_level_acc = LNLevelAccuracy()
    ln_result_acc = BxResultAccuracy()


    for mrn in mrns:
        # Radiology Report IE
        rad_nodules_manual = dataset_manual.rad.reports[mrn].nodules
        rad_nodules_llm = dataset_llm.rad.reports[mrn].nodules
        for rad_nodule_m, rad_nodule_l in zip(rad_nodules_manual, rad_nodules_llm):
            rad_loc_acc(mrn, rad_nodule_m.location, rad_nodule_l.location)
            rad_size_acc(mrn, rad_nodule_m.size, rad_nodule_l.size)
            comp_acc(mrn, rad_nodule_m.composition, rad_nodule_l.composition)
            echo_acc(mrn, rad_nodule_m.echogenicity, rad_nodule_l.echogenicity)
            shape_acc(mrn, rad_nodule_m.shape, rad_nodule_l.shape)
            margin_acc(mrn, rad_nodule_m.margin, rad_nodule_l.margin)
            foci_acc(mrn, rad_nodule_m.echogenic_foci, rad_nodule_l.echogenic_foci)
            tirads_acc(mrn, rad_nodule_m.TIRADS, rad_nodule_l.TIRADS)
        # Biopsy Report IE
        bx_nodules_manual = dataset_manual.bx.reports[mrn].nodules
        bx_nodules_llm = dataset_llm.bx.reports[mrn].nodules
        for bx_nodule_m, bx_nodule_l in zip(bx_nodules_manual, bx_nodules_llm):
            bx_loc_acc(mrn, bx_nodule_m.location, bx_nodule_l.location)
            bx_size_acc(mrn, bx_nodule_m.size, bx_nodule_l.size)
            bx_result_acc(mrn, bx_nodule_m.result, bx_nodule_l.result)
        # Lymph Node IE
        bx_ln_manual = dataset_manual.ln.reports.get(mrn)
        bx_ln_llm = dataset_llm.ln.reports.get(mrn)
        if bx_ln_manual and bx_ln_llm:
            for bx_ln_m, bx_ln_l in zip(bx_ln_manual.LN, bx_ln_llm.LN):
                ln_level_acc(mrn, bx_ln_m.level, bx_ln_l.level)
                ln_result_acc(mrn, bx_ln_m.result, bx_ln_l.result)
        elif bx_ln_manual and not bx_ln_llm:
            print(f"{mrn}: LLM missing lymph node")
        elif not bx_ln_manual and bx_ln_llm:
            print(f"{mrn}: Manual missing lymph node")
        # Nodule Matching 
        matches_manual = dataset_manual.match.reports[mrn].matches
        matches_llm = dataset_llm.match.reports[mrn].matches
        for match_m, match_l in zip(matches_manual, matches_llm):
            match_metrics(mrn, match_m, match_l)

    #print(json.dumps(bx_loc_acc.incorrect_info, indent=3))
    #print(bx_loc_acc)
    print("")

    print("\n**Rad Report IE**")
    print(f"Location Accuracy: {rad_loc_acc.accuracy*100:.1f}% ({rad_loc_acc.correct}, {rad_loc_acc.incorrect})")
    print(rad_loc_acc)
    print(f"Size Accuracy: {rad_size_acc.accuracy*100:.1f}% ({rad_size_acc.correct}, {rad_size_acc.incorrect})")
    print(rad_size_acc)
    print(f"Composition Accuracy: {comp_acc.accuracy*100:.1f}% ({comp_acc.correct}, {comp_acc.incorrect})")
    print(comp_acc)
    print(f"Echogenicity Accuracy: {echo_acc.accuracy*100:.1f}% ({echo_acc.correct}, {echo_acc.incorrect})")
    print(echo_acc)
    print(f"Shape Accuracy: {shape_acc.accuracy*100:.1f}% ({shape_acc.correct}, {shape_acc.incorrect})")
    print(shape_acc)
    print(f"Margin Accuracy: {margin_acc.accuracy*100:.1f}% ({margin_acc.correct}, {margin_acc.incorrect})")
    print(margin_acc)
    print(f"Echogenic Foci Accuracy: {foci_acc.accuracy*100:.1f}% ({foci_acc.correct}, {foci_acc.incorrect})")
    print(foci_acc)
    print(f"TIRADS Accuracy: {tirads_acc.accuracy*100:.1f}% ({tirads_acc.correct}, {tirads_acc.incorrect})")
    print(tirads_acc)

    print("\n**Bx Report IE**")
    print(f"Location Accuracy: {bx_loc_acc.accuracy*100:.1f}% ({bx_loc_acc.correct}, {bx_loc_acc.incorrect})")
    print(bx_loc_acc)
    print(f"Size Accuracy: {bx_size_acc.accuracy*100:.1f}% ({bx_size_acc.correct}, {bx_size_acc.incorrect})")
    print(bx_size_acc)
    print(f"Result Accuracy: {bx_result_acc.accuracy*100:.1f}% ({bx_result_acc.correct}, {bx_result_acc.incorrect})")
    print(bx_result_acc)
    
    print("\n**Lymph Node IE**")
    print(f"Level Accuracy: {ln_level_acc.accuracy*100:.1f}% ({ln_level_acc.correct}, {ln_level_acc.incorrect})")
    print(ln_level_acc)
    print(f"Result Accuracy: {ln_result_acc.accuracy*100:.1f}% ({ln_result_acc.correct}, {ln_result_acc.incorrect})")
    print(ln_result_acc)

    print("\n**Match**")
    print(f"TP, TN, FP, FN = {match_metrics.confusion_matrix}")
    print(match_metrics)
    TP, TN, FP, FN = match_metrics.confusion_matrix
    cm = np.array([[TP, FN], [FP, TN]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    plt.savefig("confusion")

def list_yaml_files(directory):
    """
    List all YAML files in the given directory.
    """
    pattern = os.path.join(directory, "*.yaml")
    files = sorted(glob.glob(pattern))
    return files

def select_file_from_list(files, description="file"):
    """
    Print files with numbered indices and prompt the user to select one.
    Returns the filename (without path) of the selected file.
    """
    if not files:
        print(f"No YAML files found in {description} directory.")
        return None
    print(f"\nAvailable {description} files:")
    for idx, file in enumerate(files):
        print(f"{idx}: {os.path.basename(file)}")
    while True:
        selection = input(f"Select the {description} file by entering its number: ").strip()
        try:
            index = int(selection)
            if 0 <= index < len(files):
                return os.path.basename(files[index])
            else:
                print("Invalid index, please try again.")
        except ValueError:
            print("Please enter a valid number.")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate LLM output to Manual Annoations")
    parser.add_argument("--manual", type=str,
                        help="Set manual annotation filename within 'manual_annotations' directory (e.g. 'dev_corrected.yaml')")
    parser.add_argument("--llm", type=str,
                        help="Set llm annotation filename within 'llm_results' directory (e.g. 'parsed_prompt2_dev.yaml')")
    args = parser.parse_args()

    if args.manual and args.llm:
        manual_file = args.manual
        llm_file = args.llm
    else:
        manual_files = list_yaml_files(manual_annotation_dir)
        llm_files = list_yaml_files(llm_results_dir)
        manual_file = select_file_from_list(manual_files, "manual annotation")
        llm_file = select_file_from_list(llm_files, "LLM annotation")
        if manual_file is None or llm_file is None:
            print("File selection failed. Exiting.")
            exit(1)
    
    main(manual_file, llm_file)
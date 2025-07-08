import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import torch
import pickle
import re
from torch.utils.data import Dataset
import gpustat
from ruamel.yaml import YAML

def prompt_chat_template_2(bxreports, radreports):
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                You are an expert medical report parser. Your task is to analyze a pair of reports—a radiology report and a biopsy report—and extract all the relevant details. Follow these steps carefully:
                Consider the 

                1. **Identify the Report Sections:**
                    - Step 1. locate and distinguish the parts of the input that belong to the radiology report and the biopsy report.
                
                2. **Extract Radiology Report Nodule Details (follow each step):**
                    - Step 1. identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                    - Step 2. for each nodule, extract the most detailed location, size, composition, echogenicity, shape (taller-than-wide or wider-than-tall), margin, echogenic foci, TI-RADS that can be directly found and not inferred from the report.
                    - Step 3. Change size to mm if written in cm. If size has more than one dimensions (e.g. 33 x 21 x 42 mm), report the largest value of all dimensions (e.g. 42mm).
                    - Step 4. If the information of the feature does not exist in the radiology report, report "Unknown" as the value. If the report directly mentions that there is no calcification, report echogenic foci as "no calcification".
                    - Step 5. change size to mm if written in cm. If size has more than one dimensions (e.g. 33 x 21 x 42 mm), report the largest value of all dimensions (e.g. 42mm).
                    - Step 6. Check if the the extracted information actually appear in the radiology report and assigned to correct nodule and features.
                
                3. **Extract Biopsy Report details (follow each step):**
                    - Step 1. Idnetify all samples biopsied in the biopsy report (under "FINAL DIAGNOSIS" section) and categorize them either as thyroid nodule and lymph node (e.g. level).
                    - Step 2. For each biopsied thyroid nodule, identify the location and size of the nodule under "FINAL Diagnosis" section.
                    - Step 3. For each biopsied thyroid nodule, check if more detailed location and size is available in the rest of the biopsy report.
                    - Step 4. If multiple detailed description from Step3 can be matched to nodule description from Step2, report the size and location ffrom Step 2. 
                    - Step 5. If there is only one detailed description from Step 3 can be matched to nodule description from Step 2 report the size and location from Step 3.
                    - Step 6. For each biopsied lymph node, report its biopsy result and level information only (convert arabian number) if they appear in the biopsy report.
                    - Step 7. Change size to mm if written in cm. If size has more than one dimensions (e.g. 33 x 21 x 42 mm), report the largest value of all dimensions (e.g. 42mm).
                    - Step 8. If the biopsy result is benign, report as simply "benign" but report the full description when malignent.

                4. **Identify Nodule Matching:**
                    - Step 1. For each biopsied thyroid nodule (not lymph node), check all nodules in the radiology report that can be matched based on size and location from previous steps.
                    - Step 2. For each biopsied thyroid nodule (not lymph node), if there are multiple possible matches or if it has no possible match, report the matched radiology report nodule as "Unknown" and confidence "Low".
                    - Step 3. For each biopsied thyroid noduel (not lymph node), if there is only one possible match, report the matched radiology report nodule and confidence "high"
                
                5. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level keys are "radiology_report_nodules", "biopsy_report_nodules", "biopsy_report_lymph_nodes". "nodule_matching"
                - Use exactly the field names and structure as shown below.
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                - Margin can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        size: '12 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'isoechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '3'
                    -   nodule_id: '2'
                        location: 'left mid gland'
                        size: '17 mm'
                        composition: 'mixed cystic and solid'
                        echogenicity: 'isoechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '2'
                    biopsy_report_nodules:
                    -   nodule_id: '1'
                        location: 'left mid gland'
                        size: '17 mm'
                        biopsy_result: 'benign'
                    biopsy_report_lymph_nodes:
                    -   lymph_node_id: ''
                        level: ''
                        biopsy_result: ''
                    nodule_matching:
                    -   biopsy_nodule_id: '1'
                        matched_radiology_nodule_id: '2'
                        reasoning: '' 
                        matching_confidence: ''

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_3(bxreports:list, radreports:list):
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
        bxreport = bxreport.replace("\n", "")
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                You are an expert medical report parser. Your task is to analyze a pair of reports—a radiology report and a biopsy report—and extract all the relevant details. For each task, follow all the steps carefully:
                1. **Understand the Radiology Report and Biopsy Report terms**
                    - Location refers to where a nodule is situated within the thyroid gland (e.g., in the right lobe, left lobe, or isthmus).
                    - Size is measured as the largest diameter in numerical value. If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range, report the largest value of all dimensions (e.g. 42mm).
                    - Composition refers to the internal makeup of a thyroid nodule as seen on ultrasound. It describes the relative proportions of fluid, soft tissue, and sometimes even spongiform (cyst-like spaces) elements within the nodule.
                    - Echogenicity refers to how bright or dark a nodule appears on ultrasound relative to the surrounding normal thyroid tissue.
                    - Shape refers to the geometric configuration of a nodule and can only be "taller-than-wide" or "wider-than-tall".
                    - Margins describe the edge or border of a thyroid nodule as seen on ultrasound. It can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                    - Echogenic foci are small, bright spots within a nodule on ultrasound, indicating areas that strongly reflect the ultrasound waves. If the report metions directly mentions abscent of calcification, report as "no calcifications". If the report does not mention echogenic foci of the nodule, report "Unknown"

                    - "FINAL Diagnosis" section of biopsy report contains the information of biopsied nodules.

                2. **Identify the Report Sections:**
                    - Step 1. locate and distinguish the parts of the input that belong to the radiology report and the biopsy report.
                
                3. **Extract Radiology Report Nodule Details (follow each step):**
                    You are trying to extract the nodule descriptions that you can find from radiology report. Think each step step-by-step.
                    - Step 1. identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                    - Step 2. for each nodule, extract the most detailed location, size, composition, echogenicity, shape, margin, echogenic foci, TI-RADS that can be directly found (not inferred).
                    - Step 3. If the information of the feature does not exist in the radiology report, report "Unknown" as the value.
                    - Step 4. Check if the the extracted information actually appear in the radiology report and assigned to correct nodule and features.
                
                4. **Extract Biopsy Report details (follow each step):**
                    You are trying to find the most detailed location and size for each nodule that was biopsied from the biopsy report. Think each step step-by-step.
                    - Step 1. Idnetify all samples biopsied in the biopsy report (under "FINAL DIAGNOSIS" section) and categorize them either as thyroid nodule and lymph node (e.g. level).
                    - Step 2. For each biopsied thyroid nodule, identify the location, size, and biopsy result of the nodule under "Final Diagnosis" section.
                    - Step 3. Make a list of every nodules' location and size descriptions that you can found outside of "Final Diagnosis" section with as much details as possible.
                    - Step 4. For each biopsied nodule, if multiple detailed description from Step 3 can be referred to the nodule description from Step 2, keep the original size and location from Step 2.
                              However, for each biopsied nodule if only one detailed description from Step 3 can be referred to the nodule description from Step 2, replace the nodule description with more detailed location and size from Step 3. 
                    - Step 5. For each biopsied lymph node, report its biopsy result and level information only (convert to arabian number) if they appear in the biopsy report.
                    - Step 6. If the biopsy result is benign, report as simply "benign" but report the full description when malignent.
                    - Step 7. Check if the extracted size and location information is as detailed as possible but meets the Step 4 criteria.

                5. **Identify Nodule Matching:**
                    You are trying to match each nodule from biopsy report to a nodule in radiology report if possible. Think each step step-by-step.
                    - Step 1. For each biopsied thyroid nodule (not lymph node), identity all nodules in the radiology report that has the same size or overlapping nodule location.
                    - Step 2. If there is only one possible match, report the matched radiology report nodule and confidence "high"
                              However, if there are multiple possible matches that cannot be or if it has no possible match, report the matched radiology report nodule as "Unknown" and confidence "Low".
                    - Step 3. Check if Step 2 criteria is met.

                6. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level keys are "radiology_report_nodules", "biopsy_report_nodules", "biopsy_report_lymph_nodes". "nodule_matching"
                - Use exactly the field names and structure as shown below.
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        size: '12 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'isoechoic'
                        shape: 'taller-than-wide'
                        margin: 'smooth margins'
                        echogenic_foci: 'unknown'
                        TI-RADS_Score: '3'
                    -   nodule_id: '2'
                        location: 'left mid gland'
                        size: '17 mm'
                        composition: 'mixed cystic and solid'
                        echogenicity: 'anechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '2'
                    biopsy_report_nodules:
                    -   nodule_id: '1'
                        location: 'left mid gland'
                        size: '17 mm'
                        biopsy_result: 'benign'
                    biopsy_report_lymph_nodes:
                    -   lymph_node_id: ''
                        level: ''
                        biopsy_result: ''
                    nodule_matching:
                    -   biopsy_nodule_id: '1'
                        matched_radiology_nodule_id: '2'
                        reasoning: '' 
                        matching_confidence: ''

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_4(bxreports:list, radreports:list):
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
        bxreport = bxreport.replace("\n", "")
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                **Background information the Radiology Report**
                    - Location refers to where a nodule is situated within the thyroid gland (e.g., in the right lobe, left lobe, or isthmus).
                    - Size is measured as the largest diameter in numerical value. If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range, report the largest value of all dimensions (e.g. 42mm).
                    - Composition refers to the internal makeup of a thyroid nodule as seen on ultrasound. It describes the relative proportions of fluid, soft tissue, and sometimes even spongiform (cyst-like spaces) elements within the nodule.
                    - Echogenicity refers to how bright or dark a nodule appears on ultrasound relative to the surrounding normal thyroid tissue.
                    - Shape refers to the geometric configuration of a nodule and can only be "taller-than-wide" or "wider-than-tall".
                    - Margins describe the edge or border of a thyroid nodule as seen on ultrasound. It can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                    - Echogenic foci are small, bright spots within a nodule on ultrasound, indicating areas that strongly reflect the ultrasound waves. If the report metions directly mentions abscent of calcification, report as "no calcifications". If the report does not mention echogenic foci of the nodule, report "Unknown"
                **Background information the Biopsy Report**    
                    - "Final Diagnosis" contains the information of biopsied nodules.
                    - "Clinical Information" contains clinical notes or radiology report results that can be used to provide more detailed information of the biopsied nodules.
                    - "MICROSCOPIC DESCRIPTION" contains cytological findings of the biopsied samples.
                
                You are an expert medical report parser. Your task is to analyze a pair of reports—a radiology report and a biopsy report—and extract all the relevant details. For each task, follow all the steps carefully:
                
                1. **Extract Radiology Report Nodule Details (follow each step):**
                    - Step 1. identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                    - Step 2. for each nodule, extract the most detailed location, size, composition, echogenicity, shape, margin, echogenic foci, TI-RADS that can be directly found (not inferred).
                    - Step 3. If the information of the feature does not exist in the radiology report, report "Unknown" as the value.
                    - Step 4. Check if the the extracted information actually appear in the radiology report and assigned to correct nodule and features.
                
                2. **Extract Biopsied Nodules and Lymph Nodes from Biopsy Report**
                    - Step 1. Indentify all samples biopsied in the biopsy report (under "FINAL DIAGNOSIS" section) and categorize them either as thyroid nodule and lymph node (e.g. level).
                    - Step 2. For each biopsied thyroid nodule, extract the location and biopsy result of the nodule under "Final Diagnosis" section.
                    - Step 3. For each biopsied lymph node, report its level information (convert to arabian number) and biopsy result only if they appear in the biopsy report.
                    - Step 4. If the biopsy result is benign, report as simply "benign" but report the full description when malignent.

                3. **Find More Detailed Location and Size from Biopsy Report Outside of "Final Diagnosis" Section**
                    - Step 1. Follow the rule below to decide whether to replace the location and size of the biopsied nodule with more detailed location and size extracted in this step.
                        * Rule 1: Replace the original location of biopsied nodule with more detailed location and corresponding size if there is a more detailed location and size found in biopsy report outside of "Final Diagnosis" section.
                        * Rule 2: If more than one location and size options for the original biopsied nodule, override Rule 1 and keep the original size and location.
                    - Step 2. Use the updated location and size for the final yaml output.

                4. **Identify Nodule Matching:**
                    - Step 1. For each biopsied thyroid nodule (not lymph node) after updating with more detailed location and size, identity all nodules in the radiology report that has the same size or overlapping nodule location.
                    - Step 2. If there is only one possible match, report the matched radiology report nodule and confidence "high".
                              However, if there are multiple possible matches or is no possible match, report the matched radiology report nodule as "Unknown" and confidence "Low".

                5. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level keys are "radiology_report_nodules", "biopsy_report_nodules", "biopsy_report_lymph_nodes". "nodule_matching"
                - Use exactly the field names and structure as shown below.
                - If some categories cannot be found, report as "Unknown"
                - Report the most detailed information for biopsy report nodules
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        size: '12 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'isoechoic'
                        shape: 'taller-than-wide'
                        margin: 'smooth margins'
                        echogenic_foci: 'Unknown'
                        TI-RADS_Score: '3'
                    -   nodule_id: '2'
                        location: 'left mid gland'
                        size: '17 mm'
                        composition: 'mixed cystic and solid'
                        echogenicity: 'anechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '2'
                    biopsy_report_nodules:
                    -   nodule_id: '1'
                        location: 'left mid gland'
                        size: '17 mm'
                        biopsy_result: 'benign'
                    -   nodule_id: '2'
                        location: 'right gland'
                        size: 'Unknown'
                        biopsy_result: 'Malignant - Papillary Carcinoma'
                    biopsy_report_lymph_nodes:
                    -   lymph_node_id: '1'
                        level: '6'
                        biopsy_result: 'benign'
                    nodule_matching:
                    -   biopsy_nodule_id: '1'
                        matched_radiology_nodule_id: '2'
                        reasoning: '' 
                        matching_confidence: 'High'
                    -   biopsy_nodule_id: '2'
                        matched_radiology_nodule_id: '1'
                        reasoning: '' 
                        matching_confidence: 'High'

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_5(bxreports:list, radreports:list):
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
        bxreport = bxreport.replace("\n", "")
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                **Background information the Radiology Report**
                    - Location refers to where a nodule is situated within the thyroid gland (e.g., in the right lobe, left lobe, or isthmus).
                    - Size is measured as the largest diameter in numerical value. If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range, report the largest value of all dimensions (e.g. 42mm).
                    - Composition refers to the internal makeup of a thyroid nodule as seen on ultrasound. It describes the relative proportions of fluid, soft tissue, and sometimes even spongiform (cyst-like spaces) elements within the nodule.
                    - Echogenicity refers to how bright or dark a nodule appears on ultrasound relative to the surrounding normal thyroid tissue.
                    - Shape refers to the geometric configuration of a nodule and can only be "taller-than-wide" or "wider-than-tall".
                    - Margins describe the edge or border of a thyroid nodule as seen on ultrasound. It can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                    - Echogenic foci are small, bright spots within a nodule on ultrasound, indicating areas that strongly reflect the ultrasound waves. If the report metions directly mentions abscent of calcification, report as "no calcifications". If the report does not mention echogenic foci of the nodule, report "Unknown"
                **Background information the Biopsy Report**    
                    - "FINAL DIAGNOSIS" contains the information of biopsied nodules.
                    - "CLINICAL INFORMATION" contains clinical notes or radiology report results that can be used to provide more detailed information of the biopsied nodules.
                    - "MICROSCOPIC DESCRIPTION" contains cytological findings of the biopsied samples.
                
                You are an expert medical report parser. Your task is to analyze a pair of reports—a radiology report and a biopsy report—and extract all the relevant details. For each task, follow all the steps carefully:
                
                Process 1. **Extract Radiology Report Nodule Details (follow each step):**
                    - Step 1. identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                    - Step 2. for each nodule, extract the most detailed location, size, composition, echogenicity, shape, margin, echogenic foci, TI-RADS that can be directly found (not inferred).
                    - Step 3. If the information of the feature does not exist in the radiology report, report "Unknown" as the value.
                    - Step 4. Check if the the extracted information actually appear in the radiology report and assigned to correct nodule and features.
                
                Process 2. **Extract Biopsied Nodules and Lymph Nodes from Biopsy Report**
                    - Step 1. Indentify all samples biopsied in the biopsy report (under "FINAL DIAGNOSIS" section) and categorize them either as thyroid nodule and lymph node (e.g. level).
                    - Step 2. For each biopsied thyroid nodule, extract the location and biopsy result of the nodule under "FINAL DIAGNOSIS" section.
                    - Step 3. For each biopsied lymph node, report its level information (convert to arabian number) and biopsy result only if they appear in the biopsy report.
                    - Step 4. If the biopsy result is benign, report as simply "benign" but report the full description when malignent.

                Process 3. **Extract Every Location and Size from Biopsy Report outside of "FINAL DIAGNOSIS" section**
                    - Step 1. Make a list of all nodule location and size descriptions that can be found in the Biopsy Report outside of "FINAL DIAGNOSIS" section.
                
                Process 4. **Update the Biopsied Nodules Location and Size Description
                    - Step 1. For each biopsied nodule, follow the rule below to decide whether to replace the original location and size of the biopsied nodule (process 2) with location and size found outside of "FINAL DIAGNOSIS" (process 3).
                        * Rule 1: Replace the original location of biopsied nodule with location and corresponding size found in process 3.
                        * Rule 2: If more than one descriptions from process 3 can be referred to the original biopsied nodule (process 2), override Rule 1 and keep the original size and location.
                    - Step 2. Use the updated location and size for the final yaml output.
                
                Process 5. **Identify Nodule Matching:**
                    - Step 1. For each biopsied thyroid nodule (not lymph node) after updating with more detailed location and size, identity all nodules in the radiology report that has the same size or overlapping nodule location.
                    - Step 2. If there is only one possible match, report the matched radiology report nodule and confidence "high".
                              However, if there are multiple possible matches or is no possible match, report the matched radiology report nodule as "Unknown" and confidence "Low".

                Process 6. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level keys are "radiology_report_nodules", "biopsy_report_nodules", "biopsy_report_lymph_nodes". "nodule_matching"
                - Use exactly the field names and structure as shown below.
                - If some categories cannot be found, report as "Unknown"
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        size: '12 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'isoechoic'
                        shape: 'taller-than-wide'
                        margin: 'smooth margins'
                        echogenic_foci: 'Unknown'
                        TI-RADS_Score: '3'
                    -   nodule_id: '2'
                        location: 'left mid gland'
                        size: '17 mm'
                        composition: 'mixed cystic and solid'
                        echogenicity: 'anechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '2'
                    -   nodule_id: '3'
                        location: 'right inferior gland'
                        size: '23 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'hypoechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: 'Unknown'
                    biopsy_report_nodules:
                    -   nodule_id: '1'
                        location: 'left mid gland'
                        size: '17 mm'
                        biopsy_result: 'benign'
                    -   nodule_id: '2'
                        location: 'right gland'
                        size: 'Unknown'
                        biopsy_result: 'Malignant - Papillary Carcinoma'
                    biopsy_report_lymph_nodes:
                    -   lymph_node_id: '1'
                        level: '6'
                        biopsy_result: 'benign'
                    nodule_matching:
                    -   biopsy_nodule_id: '1'
                        matched_radiology_nodule_id: '2'
                        reasoning: '' 
                        matching_confidence: 'High'
                    -   biopsy_nodule_id: '2'
                        matched_radiology_nodule_id: 'Unknown'
                        reasoning: '' 
                        matching_confidence: 'Low'

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_6(bxreports:list, radreports:list): # Prompt that ran on test set for the first time
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
        bxreport = bxreport.replace("\n", "")
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                **Background information the Radiology Report**
                    - Location refers to where a nodule is situated within the thyroid gland (e.g., in the right lobe, left lobe, or isthmus).
                    - Size is measured as the largest diameter in numerical value. If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range (e.g. up to 2 mm), report the largest dimension value (e.g. 42 mm or 2 mm).
                    - Composition refers to the internal makeup of a thyroid nodule as seen on ultrasound. It describes the relative proportions of fluid, soft tissue, and sometimes even spongiform (cyst-like spaces) elements within the nodule.
                    - Echogenicity refers to how bright or dark a nodule appears on ultrasound relative to the surrounding normal thyroid tissue.
                    - Shape refers to the geometric configuration of a nodule. The shape is either "taller-than-wide" if taller in vertical axis and "wider-than-tall" if wider in horizontal axis.
                    - Margins describe the edge or border of a thyroid nodule as seen on ultrasound. It can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                    - Echogenic foci are small, bright spots within a nodule on ultrasound, indicating areas that strongly reflect the ultrasound waves. If the report metions directly mentions abscent of calcification, report as "no calcifications". If the report does not mention echogenic foci of the nodule, report "Unknown"
                **Background information the Biopsy Report**    
                    - "FINAL DIAGNOSIS" contains the information of biopsied nodules.
                    - "CLINICAL INFORMATION" contains clinical notes or radiology report results that can be used to provide more detailed information of the biopsied nodules.
                    - "MICROSCOPIC DESCRIPTION" contains cytological findings of the biopsied samples.
                
                You are an expert medical report parser. Your task is to analyze a pair of reports—a radiology report and a biopsy report—and extract all the relevant details. For each task, follow all the steps carefully:
                
                Process 1. **Extract Radiology Report Nodule Details (follow each step):**
                    - Step 1. identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                    - Step 2. for each nodule, extract the most detailed location, size, composition, echogenicity, shape, margin, echogenic foci, TI-RADS that can be directly found (not inferred).
                    - Step 3. If the information of the feature does not exist in the radiology report, report "Unknown" as the value.
                    - Step 4. Check if the the extracted information actually appear in the radiology report and assigned to correct nodule and features.
                
                Process 2. **Extract Biopsied Nodules and Lymph Nodes from Biopsy Report**
                    - Step 1. Indentify all samples biopsied in the biopsy report (under "FINAL DIAGNOSIS" section) and categorize them either as thyroid nodule and lymph node (e.g. level).
                    - Step 2. For each biopsied thyroid nodule, extract the location and biopsy result of the nodule under "FINAL DIAGNOSIS" section. (consider possible short hand writing such as 'l' for left and 'r' for right)
                    - Step 3. For each biopsied lymph node, report its level information (convert to arabian number) and biopsy result only if they appear in the biopsy report.
                    - Step 4. If the biopsy result is benign, report as simply "benign" but report the full description when malignent.
                              (e.g. - LEFT, Benign, - Right, Malignant - Papillary Carcinoma)

                Process 3. **Extract Every Location and Size from Biopsy Report outside of "FINAL DIAGNOSIS" section**
                    - Step 1. Make a list of all nodule location and size descriptions that can be found in the Biopsy Report outside of "FINAL DIAGNOSIS" section.
                              (e.g. - left mid gland, 17 mm, - right mid gland, 12 mm, - right inferior gland, 23 mm)
                
                Process 4. **Update the Biopsied Nodules Location and Size Description**
                    - Step 1. For each biopsied nodule, follow the rule below to decide whether to replace the original location and size of the biopsied nodule (process 2) with location and size found outside of "FINAL DIAGNOSIS" (process 3).
                        * Rule 1: Replace the original location of biopsied nodule with location and corresponding size found in process 3. (e.g. update LEFT with left mid gland, 17 mm)
                        * Rule 2: If more than one descriptions from process 3 can be referred to the original biopsied nodule (process 2), override Rule 1 and keep the original size and location. (e.g. Right has two options, right mid and right inferior gland without a clear way to eliminate the candidates, so keep original, Right without size info)
                    - Step 2. Use the updated location and size for the final yaml output.
                
                Process 5. **Identify Nodule Matching:**
                    - Step 1. For each biopsied thyroid nodule (not lymph node) after updating with more detailed location and size, identity all nodules in the radiology report that has the same size or overlapping nodule location.
                    - Step 2. If there is only one possible match, report the matched radiology report nodule and confidence "high".
                              However, if there are multiple possible matches or is no possible match, report the matched radiology report nodule as "Unknown" and confidence "Low".

                Process 6. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level keys are "radiology_report_nodules", "biopsy_report_nodules", "biopsy_report_lymph_nodes". "nodule_matching"
                - Use exactly the field names and structure as the example shown below.
                - If some categories cannot be found, report as "Unknown"
                - Shape has a strict output options ["taller-than-wide, "wider-than-tall", "Unknown"]
                - Margin has a stict output options ["smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension", "Unknown".]
                - Size should report the largest dimension
                - Only Report TI-RADS Score if it is directly mentioned in the radiology report with in numerical value (between 1-5)
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        size: '12 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'isoechoic'
                        shape: 'taller-than-wide'
                        margin: 'smooth margins'
                        echogenic_foci: 'Unknown'
                        TI-RADS_Score: '3'
                    -   nodule_id: '2'
                        location: 'left mid gland'
                        size: '17 mm'
                        composition: 'mixed cystic and solid'
                        echogenicity: 'anechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '2'
                    -   nodule_id: '3'
                        location: 'right inferior gland'
                        size: '23 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'hypoechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: 'Unknown'
                    biopsy_report_nodules:
                    -   nodule_id: '1'
                        location: 'left mid gland'
                        size: '17 mm'
                        biopsy_result: 'benign'
                    -   nodule_id: '2'
                        location: 'right gland'
                        size: 'Unknown'
                        biopsy_result: 'Malignant - Papillary Carcinoma'
                    biopsy_report_lymph_nodes:
                    -   lymph_node_id: '1'
                        level: '6'
                        biopsy_result: 'benign'
                    nodule_matching:
                    -   biopsy_nodule_id: '1'
                        matched_radiology_nodule_id: '2'
                        reasoning: '' 
                        matching_confidence: 'High'
                    -   biopsy_nodule_id: '2'
                        matched_radiology_nodule_id: 'Unknown'
                        reasoning: 'Biopsied nodule has multiple possible matches (Radiology Nodule 1 and Nodule 3).' 
                        matching_confidence: 'Low'

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_Qwen1(bxreports:list, radreports:list): # Prompt that ran on test set for the first time
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
        bxreport = bxreport.replace("\n", "")
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                **Background information the Radiology Report**
                    - Location refers to where a nodule is situated within the thyroid gland (e.g., in the right lobe, left lobe, or isthmus).
                    - Size is measured as the largest diameter in numerical value. If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range (e.g. up to 2 mm), report the largest dimension value (e.g. 42 mm or 2 mm).
                    - Composition refers to the internal makeup of a thyroid nodule as seen on ultrasound. It describes the relative proportions of fluid, soft tissue, and sometimes even spongiform (cyst-like spaces) elements within the nodule.
                    - Echogenicity refers to how bright or dark a nodule appears on ultrasound relative to the surrounding normal thyroid tissue.
                    - Shape refers to the geometric configuration of a nodule. The shape is either "taller-than-wide" if taller in vertical axis and "wider-than-tall" if wider in horizontal axis.
                    - Margins describe the edge or border of a thyroid nodule as seen on ultrasound. It can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                    - Echogenic foci are small, bright spots within a nodule on ultrasound, indicating areas that strongly reflect the ultrasound waves. If the report metions directly mentions abscent of calcification, report as "no calcifications". If the report does not mention echogenic foci of the nodule, report "Unknown"
                **Background information the Biopsy Report**    
                    - "FINAL DIAGNOSIS" contains the information of biopsied nodules.
                    - "CLINICAL INFORMATION" contains clinical notes or radiology report results that can be used to provide more detailed information of the biopsied nodules.
                    - "MICROSCOPIC DESCRIPTION" contains cytological findings of the biopsied samples.
                
                You are an expert medical report parser. Your task is to analyze a pair of reports—a radiology report and a biopsy report—and extract all the relevant details. 
                Always extract the texts as is from the reports without modification. Be careful not to make any inference while extracting nodule informations.
                For each task, follow all the steps carefully:
                
                Process 1. **Extract Radiology Report Nodule Details (follow each step):**
                    - Step 1. identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                    - Step 2. for each nodule, extract the most detailed location, size, composition, echogenicity, shape, margin, echogenic foci, TI-RADS that can be directly found (not inferred).
                    - Step 3. If the information of the feature does not exist in the radiology report, report "Unknown" as the value.
                    - Step 4. Check if the the extracted information actually appear in the radiology report and not modified (as is from the report).
                    - Step 5. Check if each features information is assigned to the correct feature. Be careful not to mix features or add other features (e.g. not mix echogenicity and composition).
                
                Process 2. **Extract Biopsied Nodules and Lymph Nodes from Biopsy Report**
                    - Step 1. Indentify all samples biopsied in the biopsy report (under "FINAL DIAGNOSIS" section) and categorize them either as thyroid nodule and lymph node (e.g. level).
                    - Step 2. For each biopsied thyroid nodule, extract the location and biopsy result of the nodule under "FINAL DIAGNOSIS" section. (consider possible short hand writing such as 'l' for left and 'r' for right)
                    - Step 3. For each biopsied lymph node, report its level information (convert to arabian number) and biopsy result only if they appear in the biopsy report.
                    - Step 4. If the biopsy result is benign, report as simply "benign" but report the full description when malignent.
                        (e.g. - LEFT, Benign, 
                              - Right, Malignant - Papillary Carcinoma)

                Process 3. **Extract Every Location and Size within Biopsy Report outside of "FINAL DIAGNOSIS" section**
                    - Step 1. Make a list of all nodule location and size descriptions that can be found in the Biopsy Report outside of "FINAL DIAGNOSIS" section.
                        (e.g. - left mid gland, 17 mm, 
                              - right mid gland, 12 mm, 
                              - right inferior gland, 23 mm)
                
                Process 4.
                    - Step 1. For each biopsied nodule from process 2, list all nodules and their descriptions from process 3 that can be referred to the biopsied nodule based on the explicitly mentioned location information only.
                    Be careful not to make any assumptions.
                        (e.g. Biopsied nodule: Right -- Other Nodule descriptions: right mid gland, 12mm, - right inferior, 23 mm
                              Bipsided nodule: Left -- Other Nodule descriptions: left mid gland, 17mm)
                
                Process 5. **Update the Biopsied Nodules Location and Size Description with information from the entire biopsy report (not including radiology report)**
                    - Step 1. For each biopsied nodule, follow the rule below to decide whether to replace the original location and size of the biopsied nodule (process 2) with location and size found from (process 3) - do not consider radiology report for this task.
                        * Rule 1: The updating of information can only be done based on the excplicitly mentioned location information or explicit mention of biospy (e.g. subjected to FNA, recommended for FNA)
                        * Rule 2: Order of the nodule in the biopsy report does not mean anything. If multiple nodules with same location was biopsied  (e.g. right and right) it does not inherently mean that the first right biopsied nodule (process 2) belongs to the first more detailed nodule description (from process 3) since both permutations are possible)
                        * Rule 3: If more than one descriptions from process 3 can be referred to the original biopsied nodule (process 2) based on process 4, keep the original size and location without updating. (e.g. Right has two options, right mid and right inferior gland without a clear way to eliminate the candidates, so keep original, Right without size info)
                        * Rule 5: If more than one biopsied nodule can be referred to the same nodule description from process 3, keep the original size and location without updating. (e.g. There are two right biopsied nodules, and there is one option - right inferior, 13mm, you cannot associate it to any of the biopsied nodule since it is ambiguous)
                        * Rule 4: If only one descriptions from process 3 can be referred to the original biopsied nodule (process 2) based on process 4 and if the description is more detailed than the orginal description, replace the original location of biopsied nodule with location and corresponding size found in process 3. (e.g. update LEFT with left mid gland, 17 mm)
                    - Step 2. Use the updated location and size for the final yaml output.
                
                Process 6. **Identify Nodule Matching:**
                    Definition of match: Two nodules, each from radiology and biopsy reports can be considered identical.
                    - Step 1. For each biopsied thyroid nodule (not lymph node) identify all nodules in the radiology report that can be matches based on the updated biopsied nodule desciption.
                    - Step 2. If there are multiple possible radiology report nodules that can be matched to the biopsied nodule or if there is no possible match, report the matched radiology report nodule as "Unknown"
                    - Step 3. If there is only one possible match, report the matched radiology report nodule.
                    - Step 4. Evaluate your matching and report the confidence level of your decision as "Low" or "High"

                Process 7. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level keys are "radiology_report_nodules", "biopsy_report_nodules", "biopsy_report_lymph_nodes". "nodule_matching"
                - Use exactly the field names and structure as the example shown below.
                - If some categories cannot be found, report as "Unknown"
                - Echogenicity and Composition can cantain prepositions or adjectives
                - Shape has a strict output options ["taller-than-wide, "wider-than-tall", "Unknown"]
                - Margin has a stict output options ["smooth", "ill-defined", "lobulated", "irregular", "irregular or lobulated", "extra-thyroidal extension", "Unknown".]
                - Size should report the largest dimension
                - Only Report TI-RADS Score if it is directly mentioned in the radiology report with in numerical value (between 1-5)
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        size: '12 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'isoechoic'
                        shape: 'taller-than-wide'
                        margin: 'smooth margins'
                        echogenic_foci: 'Unknown'
                        TI-RADS_Score: '3'
                    -   nodule_id: '2'
                        location: 'left mid gland'
                        size: '17 mm'
                        composition: 'mixed cystic and solid'
                        echogenicity: 'anechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '2'
                    -   nodule_id: '3'
                        location: 'right inferior gland'
                        size: '23 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'hypoechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: 'Unknown'
                    biopsy_report_nodules:
                    -   nodule_id: '1'
                        location: 'left mid gland'
                        size: '17 mm'
                        biopsy_result: 'benign'
                    -   nodule_id: '2'
                        location: 'right gland'
                        size: 'Unknown'
                        biopsy_result: 'Malignant - Papillary Carcinoma'
                    biopsy_report_lymph_nodes:
                    -   lymph_node_id: '1'
                        level: '6'
                        biopsy_result: 'benign'
                    nodule_matching:
                    -   biopsy_nodule_id: '1'
                        matched_radiology_nodule_id: '2'
                        reasoning: ''
                        matching_confidence: 'High'
                    -   biopsy_nodule_id: '2'
                        matched_radiology_nodule_id: 'Unknown'
                        reasoning: 'Biopsied nodule has multiple possible matches (Radiology Nodule 1 and Nodule 3).' 
                        matching_confidence: 'Low'

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_Qwen2(bxreports:list, radreports:list): # Prompt that ran on test set for the first time
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
        bxreport = bxreport.replace("\n", "")
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                **Background information the Radiology Report**
                    - Location refers to where a nodule is situated within the thyroid gland (e.g., in the right lobe, left lobe, or isthmus).
                    - Size is measured as the largest diameter in numerical value. If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range (e.g. up to 2 mm), report the largest dimension value (e.g. 42 mm or 2 mm).
                    - Composition refers to the internal makeup of a thyroid nodule as seen on ultrasound. It describes the relative proportions of fluid, soft tissue, and sometimes even spongiform (cyst-like spaces) elements within the nodule.
                    - Echogenicity refers to how bright or dark a nodule appears on ultrasound relative to the surrounding normal thyroid tissue.
                    - Shape refers to the geometric configuration of a nodule. The shape is either "taller-than-wide" if taller in vertical axis and "wider-than-tall" if wider in horizontal axis.
                    - Margins describe the edge or border of a thyroid nodule as seen on ultrasound. It can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                    - Echogenic foci are small, bright spots within a nodule on ultrasound, indicating areas that strongly reflect the ultrasound waves. If the report metions directly mentions abscent of calcification, report as "no calcifications". If the report does not mention echogenic foci of the nodule, report "Unknown"
                **Background information the Biopsy Report**    
                    - "FINAL DIAGNOSIS" contains the information of biopsied nodules.
                    - "CLINICAL INFORMATION" contains clinical notes or radiology report results that can be used to provide more detailed information of the biopsied nodules.
                    - "MICROSCOPIC DESCRIPTION" contains cytological findings of the biopsied samples.
                
                You are an expert medical report parser. Your task is to analyze a pair of reports—a radiology report and a biopsy report—and extract all the relevant details. 
                Always extract the texts as is from the reports without modification. Be careful not to make any inference while extracting nodule informations.
                For each task, follow all the steps carefully:
                
                Task 1. **Extract Radiology Report Nodule Details**
                You need to extract all the details of all thyroid nodules in the radiology report obeying all the instrcutions below.
                    Instructions:
                        1) Lymph nodes, parthyroid nodules or nodules that are grouped together such as "multiple nodules" are not part of thyroid nodule you need to extract information of.
                        2) Do not make inferencem and only extract text directly as it appears in the radiology report.
                        3) If the information of the feature does not exist in the radiology report, report "Unknown" as the value.
                        2) You need to extract location, size, composition, echogenicity, shape, margin, echogenic foci and TI-RADS score.
                After the extraction, check if the the extracted information actually appear in the radiology report and not modified (as is from the report).
                Then check if each features information is assigned to the correct feature. Be careful not to mix features or add other features (e.g. not mix echogenicity and composition).
                
                Task 2. **Extract Biopsied Nodules Information from Biopsy Report**
                
                You need to identify the thyroid nodules (not lymph nodes which might be also expressed as 'level') that were biopsied in the current biopsy report and their biopsy result. 
                The current biopsied nodules and their biopsy results are located under "FINAL DIAGNOSIS" section along with the biopsy result.
                
                In addition to identifying the biopsied nodule, you need to extract the most detailed location and size information for each of the biopsied nodule following the instruction below. 
                For example, let's say the locations of two nodules described under "Final Diagnosis section" are each Left and Right. However, if you find more detailed location and size information somewhere else in the biopsy report that correspond to those nodule,
                you should update the origin Left and Right to something like 'left mid' and 'right inferior'.
                
                However, if you have more than one location and size information somwhere else in the biopsy report that correspond to those nodule, you should keep the original less detailed location. 
                For example, if there are two nodule locations described, 'right mid' and 'right inferior', both can be corresponded to the original location description 'Right'. In this case, you should keep the original location because you are not sure which one is the correct one to be matched.
                Another example is that if there are two nodule locations 'left inferior 13mm' and 'left inferior 16mm' in the biopsy report outside of "FINAL DIAGNOSIS" section, there are two possible option for original location 'LEFT'. Therefore you should keep the original location and size.

                    Instructions:
                        1) Important location informations are right, left, and isthmus. If you find short handed writing such as R, L you can assume they are Right and Left.
                        2) You need to report some kind of location whether it is detailed or not. However, you do not have to report the size if you are not 100% confident.
                        2) If the biopsy result is benign, report as simply "benign" but report the full description when malignent. (e.g. - LEFT, Benign, - Right, Malignant - Papillary Carcinoma)
                        3) If biopsy report directly mentions that a specific nodule was biopsied (e.g. subjected to FNA) you can rule out other possible candidates for more detailed location and size.
                        4) Order of the nodule in the biopsy report does not have any relation ship to the order of the nodule described in the "Final Diagnosis". Do not use order as clue to draw connection.
                
                If the size information cannot be assigned with confidence, return as "unknown"
                
                Task 3 **Extract Lymph Node Information from Biopsy Report**

                You need to extract location and biopsy result of the lymph nodes (e.g. level) not thyroid nodules from the biopsy report under "Final Diagnosis" section.
                Report empty list if no lymph node was biopsied in the current biopsy report.

                Task 4. **Identify Nodule Matching:**

                You need to match a nodule from radiology report to a nodule in the biopsy report. This time you should use the entire radiology report and the extracted detailed information of the biopsied nodule from Task 2.
                Nodules in the biopsy report are the base. For each nodule in the biopsy report, you should think of all the nodules from the radiology report can matches the detailed location and size description of the biopsied nodule from Task 2.
                
                    Instructions:
                        1) All matching should be based on exact location and size. 
                        2) If the size is different, it is not a match. (e.g. left inferior 12 mm biopsied nodule cannot be matched to left inferior 14 mm radiology report nodule)
                        3) Unlike size, that require exact matching, if the location of the biopsied nodule is less detailed (e.g. left 4 mm) and the radiology report nodule has more detailed location (e.g. left inferior 4 mm), they can be a valid match
                        4) If more than one radioloy report nodule can be matched to one biopsied nodule, report the radiology report nodule match as unknown.
                        5) Report the confidence of your matching performance as "High" or "Low". If you made assumption on the way of matching, report "Low" confidence.

                Final Output. 
                **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level keys are "radiology_report_nodules", "biopsy_report_nodules", "biopsy_report_lymph_nodes". "nodule_matching"
                - Use exactly the field names and structure as the example shown below.
                - If some categories cannot be found, report as "Unknown"
                - Echogenicity and Composition can cantain prepositions or adjectives
                - Shape has a strict output options ["taller-than-wide, "wider-than-tall", "Unknown"]
                - Margin has a stict output options ["smooth", "ill-defined", "lobulated", "irregular", "irregular or lobulated", "extra-thyroidal extension", "Unknown".]
                - Size should report the largest dimension
                - Do not report individual feature TI-RADS rating in the parenthesis (e.g. (1), (2))
                - Only Report TI-RADS Score if it is directly mentioned in the radiology report with in numerical value (between 1-5)
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        size: '12 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'isoechoic'
                        shape: 'taller-than-wide'
                        margin: 'smooth margins'
                        echogenic_foci: 'Unknown'
                        TI-RADS_Score: '3'
                    -   nodule_id: '2'
                        location: 'left mid gland'
                        size: '17 mm'
                        composition: 'mixed cystic and solid'
                        echogenicity: 'anechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '2'
                    -   nodule_id: '3'
                        location: 'right inferior gland'
                        size: '23 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'hypoechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: 'Unknown'
                    biopsy_report_nodules:
                    -   nodule_id: '1'
                        location: 'left mid gland'
                        size: '17 mm'
                        biopsy_result: 'benign'
                    -   nodule_id: '2'
                        location: 'right gland'
                        size: 'Unknown'
                        biopsy_result: 'Malignant - Papillary Carcinoma'
                    biopsy_report_lymph_nodes:
                    -   lymph_node_id: '1'
                        level: '6'
                        biopsy_result: 'benign'
                    nodule_matching:
                    -   biopsy_nodule_id: '1'
                        matched_radiology_nodule_id: '2'
                        reasoning: '' 
                        matching_confidence: 'High'
                    -   biopsy_nodule_id: '2'
                        matched_radiology_nodule_id: 'Unknown'
                        reasoning: 'Biopsied nodule has multiple possible matches (Radiology Nodule 1 and Nodule 3).' 
                        matching_confidence: 'Low'

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_7(bxreports:list, radreports:list): # Improving Biopsy Loc performanc
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
        bxreport = bxreport.replace("\n", "")
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                **Background information the Radiology Report**
                    - Location refers to where a nodule is situated within the thyroid gland (e.g., in the right lobe, left lobe, or isthmus).
                    - Size is measured as the largest diameter in numerical value. If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range (e.g. up to 2 mm), report the largest dimension value (e.g. 42 mm or 2 mm).
                    - Composition refers to the internal makeup of a thyroid nodule as seen on ultrasound. It describes the relative proportions of fluid, soft tissue, and sometimes even spongiform (cyst-like spaces) elements within the nodule.
                    - Echogenicity refers to how bright or dark a nodule appears on ultrasound relative to the surrounding normal thyroid tissue.
                    - Shape refers to the geometric configuration of a nodule. The shape is either "taller-than-wide" if taller in vertical axis and "wider-than-tall" if wider in horizontal axis.
                    - Margins describe the edge or border of a thyroid nodule as seen on ultrasound. It can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                    - Echogenic foci are small, bright spots within a nodule on ultrasound, indicating areas that strongly reflect the ultrasound waves. If the report metions directly mentions abscent of calcification, report as "no calcifications". If the report does not mention echogenic foci of the nodule, report "Unknown"
                **Background information the Biopsy Report**    
                    - "FINAL DIAGNOSIS" contains the information of biopsied nodules.
                    - "CLINICAL INFORMATION" contains clinical notes or radiology report results that can be used to provide more detailed information of the biopsied nodules.
                    - "MICROSCOPIC DESCRIPTION" contains cytological findings of the biopsied samples.
                
                You are an expert medical report parser. Your task is to analyze a pair of reports—a radiology report and a biopsy report—and extract all the relevant details. For each task, follow all the steps carefully.
                
                Process 1. **Extract Radiology Report Nodule Details (follow each step):**
                    - Step 1. identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                    - Step 2. for each nodule, extract the most detailed location, size, composition, echogenicity, shape, margin, echogenic foci, TI-RADS that can be directly found (not inferred).
                    - Step 3. If the information of the feature does not exist in the radiology report, report "Unknown" as the value.
                    - Step 4. Check if the the extracted information actually appear in the radiology report and assigned to correct nodule and features.
                    e.g.**radiology_report_nodules**
                        -   nodule_id: '1'
                            location: 'right mid gland'
                            size: '12 mm'
                            composition: 'solid or almost completely (>95%) solid'
                            echogenicity: 'isoechoic'
                            shape: 'taller-than-wide'
                            margin: 'smooth margins'
                            echogenic_foci: 'Unknown'
                            TI-RADS_Score: '3'
                        -   nodule_id: '2'
                            location: 'left mid gland'
                            size: '17 mm'
                            composition: 'mixed cystic and solid'
                            echogenicity: 'anechoic'
                            shape: 'wider-than-tall'
                            margin: 'smooth margins'
                            echogenic_foci: 'no calcifications'
                            TI-RADS_Score: '2'
                        -   nodule_id: '3'
                            location: 'right inferior gland'
                            size: '23 mm'
                            composition: 'solid or almost completely (>95%) solid'
                            echogenicity: 'hypoechoic'
                            shape: 'wider-than-tall'
                            margin: 'smooth margins'
                            echogenic_foci: 'no calcifications'
                            TI-RADS_Score: 'Unknown'
                Process 2. **Extract Biopsied Nodules and Lymph Nodes from Biopsy Report**
                    - Step 1. Indentify all samples biopsied in the biopsy report (under "FINAL DIAGNOSIS" section) and categorize them either as thyroid nodule and lymph node (e.g. level).
                    - Step 2. For each biopsied thyroid nodule, extract the location and biopsy result of the nodule under "FINAL DIAGNOSIS" section. (consider possible short hand writing such as 'l' for left and 'r' for right)
                    - Step 3. For each biopsied lymph node, report its level information (convert to arabian number) and biopsy result only if they appear in the biopsy report.
                    - Step 4. If the biopsy result is benign, report as simply "benign" but report the full description when malignent.
                    e.g. **Biopsied Nodule Description**
                        - biopsied_nodule 1: 
                            - Location: LEFT
                            - Size: Unknown
                            - Biopsy Result: Benign
                        - biopsied_nodule 2: 
                            - Location: Right
                            - Size: Unkown
                            - Biopsy Result: Malignant - Papillary Carcinoma

                Process 3. **Extract Every Location and Size from Biopsy Report outside of "FINAL DIAGNOSIS" section**
                    - Step 1. Identify all nodule location and size descriptions that can be found in the Biopsy Report outside of "FINAL DIAGNOSIS" section and write it down for reference in process 4.
                    e.g. **All Nodue Descriptions Outside of "Final Diagnosis"
                        - nodule_description 1: 
                            - Location: left mid gland
                            - Size: 17 mm 
                        - nodule_description 2: 
                            - Location: right mid gland
                            - Size: 12 mm
                        - nodule_description 3: 
                            - Location: right inferior gland
                            - Size: 23 mm
                    - Step 2. Indentify nodules that biopsy report directly mentions they were biopsied (e.g. suggested for FNA)
                
                Process 4. **Update the Biopsied Nodules Location and Size Description**
                    - Step 1. Remind yourself to be as conservative as possible in updating. If there are not enough direct evidence to update, do not update. Do not make assumption or indirect inference.
                    - Step 2. If there is more detailed location or size found in biopsy report (not radiology report) outside of "FINAL DIAGNOSIS" (process 3) that can be referred to a biopsied nodule, replace the original descript of the biopsied nodule (process 2) with more detailed description (procee 3)
                    - Step 3. Check if more than one possible location from process 3 that can be matched to a single biopsied nodule. If there is not enough information within the biopsy report to rule out other candidates, keep the origin descriptions (process 2), unless biospy report directly mentions that the nodule was suggested for biopsy (FNA) or biopsied.
                    e.g. **Updated Biopsy Nodule**
                        - biopsied_nodule 1: (replaced original description from process 2 with more detailed nodule description from process 3 since there was only one possible association)
                            - Location: left mid gland
                            - Size: 17 mm
                            - Biopsy Result: Benign
                        - biopsied_nodule 2: (kept original description from process 2 since there are two more detailed nodule descriptions from process 3 that can be referred to "Right" but there is not enough information to rule out one or the other) 
                            - Location: Right
                            - Size: Unknown
                            - Biopsy Result: Malignant - Papillary Carcinoma
                    - Step 4. Use the updated location and size for the final yaml output's biopsy_report_nodules
                
                Process 5. **Identify Nodule Matching:**
                    - Step 1. For each biopsied thyroid nodule (not lymph node) after updating with more detailed location and size, identity all nodules in the radiology report that has the same size or overlapping nodule location.
                    - Step 2. If there is only one possible match, report the matched radiology report nodule and confidence "high".
                              However, if there are multiple possible matches or is no possible match, report the matched radiology report nodule as "Unknown" and confidence "Low".

                Process 6. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level keys are "radiology_report_nodules", "biopsy_report_nodules", "biopsy_report_lymph_nodes". "nodule_matching"
                - Use exactly the field names and structure as the example shown below.
                - If some categories cannot be found, report as "Unknown"
                - Shape has a strict output options ["taller-than-wide, "wider-than-tall", "Unknown"]
                - Margin has a stict output options ["smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension", "Unknown".]
                - Size should report the largest dimension
                - Only Report TI-RADS Score if it is directly mentioned in the radiology report with in numerical value (between 1-5)
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        size: '12 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'isoechoic'
                        shape: 'taller-than-wide'
                        margin: 'smooth margins'
                        echogenic_foci: 'Unknown'
                        TI-RADS_Score: '3'
                    -   nodule_id: '2'
                        location: 'left mid gland'
                        size: '17 mm'
                        composition: 'mixed cystic and solid'
                        echogenicity: 'anechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '2'
                    -   nodule_id: '3'
                        location: 'right inferior gland'
                        size: '23 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'hypoechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: 'Unknown'
                    biopsy_report_nodules:
                    -   nodule_id: '1'
                        location: 'left mid gland'
                        size: '17 mm'
                        biopsy_result: 'benign'
                    -   nodule_id: '2'
                        location: 'right gland'
                        size: 'Unknown'
                        biopsy_result: 'Malignant - Papillary Carcinoma'
                    biopsy_report_lymph_nodes:
                    -   lymph_node_id: '1'
                        level: '6'
                        biopsy_result: 'benign'
                    nodule_matching:
                    -   biopsy_nodule_id: '1'
                        matched_radiology_nodule_id: '2'
                        reasoning: '' 
                        matching_confidence: 'High'
                    -   biopsy_nodule_id: '2'
                        matched_radiology_nodule_id: 'Unknown'
                        reasoning: 'Biopsied nodule has multiple possible matches (Radiology Nodule 1 and Nodule 3).' 
                        matching_confidence: 'Low'

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_2(bxreports, radreports, llm_yamls):
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports, llm_yamls):
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
            You are a expert annotator reviewing the annotation and evaluating whether each section of the annotation was done correctly based on the rules below. 

            **Radiology Report Nodules**
            Rule 1. Did all of the nodule features extracted as is from the text (copy-and-paste)?
            Rule 2. Did you create 

            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Radiology_Report: {radreport},
            Annotation: {llm_yamls}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts


def prompt_chat_template_rad_only(radreports:list): #separate each task
    prompts = []
    for radreport in radreports:
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                **Background information of the thyroid nodules**
                    - thyroid nodules are nodules includes all nodules described in the report that are not parathyroid nodules or lymph nodes (e.g. level 6, lymph, etc.)
                **Background information the Radiology Report**
                    - Location refers to where a nodule is situated within the thyroid gland (e.g., in the right lobe, left lobe, or isthmus).
                    - Size is measured as the largest diameter in numerical value. If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range (e.g. up to 2 mm), report the largest dimension value (e.g. 42 mm or 2 mm).
                    - Composition refers to the internal makeup of a thyroid nodule as seen on ultrasound. It describes the relative proportions of fluid, soft tissue, and sometimes even spongiform (cyst-like spaces) elements within the nodule.
                    - Echogenicity refers to how bright or dark a nodule appears on ultrasound relative to the surrounding normal thyroid tissue.
                    - Shape refers to the geometric configuration of a nodule. The shape is either "taller-than-wide" if taller in vertical axis and "wider-than-tall" if wider in horizontal axis.
                    - Margins describe the edge or border of a thyroid nodule as seen on ultrasound. It can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                    - Echogenic foci are small, bright spots within a nodule on ultrasound. If the report metions directly mentions abscent of calcification, report as "no calcifications". If the report does not mention echogenic foci of the nodule, report "Unknown"

                You are an expert medical report parser. Your task is to analyze a radiology report and extract all the relevant details. For each task, follow all the steps carefully:
                
                Process 1. **Extract Radiology Report Nodule Details (follow each step):**
                    - Step 1. identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                    - Step 2. for each nodule, extract the most detailed location, size, composition, echogenicity, shape, margin, echogenic foci, TI-RADS that can be directly found (not inferred).
                    - Step 3. If the information of the feature does not exist in the radiology report, report "Unknown" as the value.
                    - Step 4. Check if the the extracted information actually appear in the radiology report.
                    - Step 5. Check if the extracted information follows the format rules

                Format Rules:
                    - Rule 1: If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range (e.g. up to 2 mm), report the largest dimension value (e.g. 42 mm or 2 mm).
                    - Rule 2: If a feature of a nodule does not directly appear in the radiology report, the extracted value should be "unknown"
                    - Rule 3: Margin has a stict output options ["smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension", "Unknown".]
                    - Rule 4: Shape has a strict output options ["taller-than-wide, "wider-than-tall", "Unknown"]. Do not confuse between taller-than-wide and wider-than-tall since they are completely opposite.
                    - Rule 5: TI-RADS score is extracted if a numeral TI-RADS score (between 1-5) is explicitly mentioned.
                
                Process 2. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level key "radiology_report_nodules" and following features as second-level keys.
                - Use exactly the field names and structure as the example shown below.
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        size: '12 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'isoechoic'
                        shape: 'taller-than-wide'
                        margin: 'smooth margins'
                        echogenic_foci: 'Unknown'
                        TI-RADS_Score: '3'
                    -   nodule_id: '2'
                        location: 'left mid gland'
                        size: '17 mm'
                        composition: 'mixed cystic and solid'
                        echogenicity: 'anechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: '2'
                    -   nodule_id: '3'
                        location: 'right inferior gland'
                        size: '23 mm'
                        composition: 'solid or almost completely (>95%) solid'
                        echogenicity: 'hypoechoic'
                        shape: 'wider-than-tall'
                        margin: 'smooth margins'
                        echogenic_foci: 'no calcifications'
                        TI-RADS_Score: 'Unknown'

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_bx_only(bxreports:list): #separate each task
    prompts = []
    for bxreport in bxreports:
        bxreport = bxreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                **Background information of the thyroid nodules**
                    - thyroid nodules are nodules includes all nodules described in the report that are not parathyroid nodules or lymph nodes (e.g. level 6, lymph, etc.)
                **Background information the Biopsy Report**    
                    - "FINAL DIAGNOSIS" contains the information of biopsied nodules including all thyroid nodules, parathyroid nodules, and lymph node.
                    - "CLINICAL INFORMATION" contains clinical notes or radiology report results that can be used to provide more detailed information of the biopsied nodules.
                    - "MICROSCOPIC DESCRIPTION" contains cytological findings of the biopsied samples.
                
                You are an expert medical report parser. Your task is to analyze a biopsy report and extract all the relevant details. For each task, follow all the steps carefully:
                
                Process 1. **Extract Biopsied Nodules and Lymph Nodes from Biopsy Report**
                    - Step 1. Indentify all samples biopsied in the biopsy report (under "FINAL DIAGNOSIS" section) and categorize them either as thyroid nodule and lymph node (e.g. level).
                    - Step 2. For each biopsied thyroid nodule, extract the location and biopsy result of the nodule under "FINAL DIAGNOSIS" section. (consider possible short hand writing such as 'l' for left and 'r' for right)
                    - Step 3. For each biopsied lymph node, report its level information (convert to arabian number) and biopsy result only if they appear in the biopsy report. (e.g. level 4, benign)
                    - Step 4. If the biopsy result is benign, report as simply "benign" but report the full description when malignent.
                              (e.g. - LEFT, Benign, - Right, Malignant - Papillary Carcinoma)

                Process 2. **Extract Every Location and Size from Biopsy Report outside of "FINAL DIAGNOSIS" section**
                    - Step 1. Make a list of all nodule location and size descriptions that can be found in the Biopsy Report outside of "FINAL DIAGNOSIS" section with as much detail.
                              (e.g. - left mid gland, 17 mm, - right mid gland, 21 mm, - right inferior gland, 8 mm)
                
                Process 3. **Update the Biopsied Nodules Location and Size Description
                    - Step 1. For each biopsied nodule, follow the rule below to decide whether to replace the original location and size of the biopsied nodule (process 2) with location and size found outside of "FINAL DIAGNOSIS" (process 3).
                        * Rule 1: Identify all nodule descritions found in process 3 that matches the description of process 2. (e.g.1 LEFT: left mid gland, 17 mm) (e.g.2 Right: right mid, 21 mm / right inferior gland, 8 mm)
                        * Rule 2: If more than one descriptions from process 3 can be referred to the original biopsied nodule (process 2), keep the original size and location. (e.g. Right has two options, right mid and right inferior gland without a clear way to eliminate the candidates, so keep original, Right without size info)
                        * Rule 3: If only one description from process 3 can be refeered to the original biopsied nodule, and if description in process 3 has more detailed description of the nodule, 
                                replace the detail in process 2 with the detail in process 3. (e.g. Replce LEFT with left mid gland and size of 17 mm)

                Process 4. **Structure the Output in YAML:**
                - Use the updated location and size for the final yaml output.
                - The final output should be a YAML document with a top-level keys are  "biopsy_report_nodules", and "biopsy_report_lymph_nodes".
                - Use exactly the field names and structure as the example shown below.
                - If some categories cannot be found, report as "Unknown"
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    biopsy_report_nodules:
                    -   nodule_id: '1'
                        location: 'left mid gland'
                        size: '17 mm'
                        biopsy_result: 'benign'
                    -   nodule_id: '2'
                        location: 'right gland'
                        size: 'Unknown'
                        biopsy_result: 'Malignant - Papillary Carcinoma'
                    biopsy_report_lymph_nodes:
                    -   lymph_node_id: '1'
                        level: '4'
                        biopsy_result: 'benign'

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Let's think step-by-step
            """}
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_match_only(bxreports:list, radreports:list, bx_yamls:list, rad_yamls): # Prompt that ran on test set for the first time
    prompts = []
    for bxreport, radreport, bx_yaml, rad_yaml in zip(bxreports, radreports, bx_yamls, rad_yamls):
        bxreport = bxreport.replace("\n", "")
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""                
                You are an expert annotator. Your task is to analyze a pair of reports and match biopsied nodules to the nodules described in the radiology report. For each task, follow all the steps carefully:
                
                Process 1. **Identify Nodule Matching:**
                    - Step 1. Check the numer and information of the biopsied nodules and their description from Biopsy_Report_Nodule_Info
                    - Step 2. Use size and location information from Biospy_Report_Nodule_Info to identify all nodules in Radiology_Report_Nodule_Info that can be matched to each nodule in Biopsy_Report_Nodule_Info)
                    - Step 3. If there are multiple possible matche check if additional information from Biopsy Report and Radiology Report can be used to identify the one nodule from radiology report that matches the biopsied nodules
                    - Step 4. If the possible matches cannot be reduced after Step 3 or if there is no possible match, report the matched radiology report nodule as "Unknown" and confidence "Low".
                    - Step 4. If there is only one possible match, report the matched radiology report nodule and confidence "high".

                Process 2. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level key "nodule_matching".
                - Nodule ID should always be numerical. Convert Alphabet ID to corresponding numerical ID. (e.g. 'A' to '1', 'B' to '2')
                
                **Example YAML Structure:**
                    nodule_matching:
                    -   biopsy_nodule_id: '1'
                        matched_radiology_nodule_id: '2'
                        reasoning: '' 
                        matching_confidence: 'High'
                    -   biopsy_nodule_id: '2'
                        matched_radiology_nodule_id: 'Unknown'
                        reasoning: 'Biopsied nodule has multiple possible matches (Radiology Nodule 1 and Nodule 3).' 
                        matching_confidence: 'Low'

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Biopsy_Report: {bxreport},
            Biospy_Report_Nodule_Info: {bx_yaml}
            Radiology_Report: {radreport}
            Radiology_Report_Nodule_Info: {rad_yaml}

            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def prompt_chat_template_ashwath(radreports:list): #separate each task
    prompts = []
    for radreport in radreports:
        radreport = radreport.replace("\n", "")
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                **Background information of the thyroid nodules**
                    - thyroid nodules are nodules includes all nodules described in the report that are not parathyroid nodules or lymph nodes (e.g. level 6, lymph, etc.)
                **Background information the Radiology Report**
                    - Location refers to where a nodule is situated within the thyroid gland (e.g., in the right lobe, left lobe, or isthmus).
                    - Size is measured as the largest diameter in numerical value. If size has more than one dimensions (e.g. 33 x 21 x 42 mm) or described as a range (e.g. up to 2 mm), report the largest dimension value (e.g. 42 mm or 2 mm).
                    - Composition refers to the internal makeup of a thyroid nodule as seen on ultrasound. It describes the relative proportions of fluid, soft tissue, and sometimes even spongiform (cyst-like spaces) elements within the nodule.
                    - Echogenicity refers to how bright or dark a nodule appears on ultrasound relative to the surrounding normal thyroid tissue.
                    - Shape refers to the geometric configuration of a nodule. The shape is either "taller-than-wide" if taller in vertical axis and "wider-than-tall" if wider in horizontal axis.
                    - Margins describe the edge or border of a thyroid nodule as seen on ultrasound. It can take a value from "smooth", "ill-defined", "lobulated", "irregular", "extra-thyroidal extension". If not mentioned, return "Unknown".
                    - Echogenic foci are small, bright spots within a nodule on ultrasound. If the report metions directly mentions abscent of calcification, report as "no calcifications". If the report does not mention echogenic foci of the nodule, report "Unknown"

                You are an expert medical report parser. Your task is to analyze a radiology report and extract the location of all nodules and sentences that describe each of the nodule. For each task, follow all the steps carefully:
                Copy and and paste all the setences that describe the nodule witout modification.
                The nodule description could follow either ATA guideline or TI-RADS format.

                - Step 1. Identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                - Step 2. For each nodule, extract the most detailed location.
                - Step 3. Copy and paste all the sentences that describe each nodules.
                - Step 4. Double check if the sentences assigned to a nodule does not belong to other nodules, parathyroid nodules or lymph nodes.
                
                Process 2. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level key "radiology_report_nodules" and following features as second-level keys.
                - Use exactly the field names and structure as the example shown below.
                - Nodule ID should always be numerical.
                
                **Example YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: '1'
                        location: 'right mid gland'
                        description sentences: "34 mm right mid gland in nodule, solid or almost completely (>95%) solid (2), isoechoic (1), wider-than-tall (0), smooth margins (0), with no calcifications (0), TI-RADS 3 (Total 3 points) , previously measuring 26 mm."
                    -   nodule_id: '2'
                        location: 'superior right gland'
                        description sentences: "There is a solitary nodule measuring 7 x 5 x 7 mm in the superior right gland demonstrating heterogeneous echotexture with multiple punctate echogenic foci suspicious for microcalcifications. The margins of the nodule are somewhat poorly defined and there is no significant vascularity. ATA 2015 high suspicion pattern."

                Do not modify the information when extracting them from the reports and extract as is from the reports.
                
                Think step-by-step.
            """},

            {"role": "user",
            "content": f"""
            Radiology_Report: {radreport}
            Let's think step-by-step
            """} 
        ]
        prompts.append(messages)
    return prompts

def generate_text_with_chat_template(tokenizer, model, batch_messages, do_sample=False, temperature=None, top_p=None, top_k=None, max_tokens=2048):
    
    #tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.apply_chat_template(
        batch_messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt", 
        return_dict=True,
        padding=True,            
        truncation=True,
        max_tokens=128000        
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    processed_texts = []
    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        processed = decoded.split("assistant")[-1].strip().strip("`").strip('json')
        processed_texts.append(processed)
        
    return processed_texts
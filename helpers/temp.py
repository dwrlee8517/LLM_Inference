def prompt_chat_template(bxreports, radreports):
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
    # Define the messages
        messages = [
            {"role": "system",
            "content": f"""
                You are an expert medical report parser. Your task is to analyze a pair of reports—a radiology report and a biopsy report—and extract all the relevant details. Follow these steps carefully:

                1. **Identify the Report Sections:**
                    - Step 1. locate and distinguish the parts of the input that belong to the radiology report and the biopsy report.
                
                2. **Extract Radiology Report Nodule Details (follow each step):**
                    - Step 1. identify all nodules in the report. Do not identify lymph node, parathyroid nodule, or nodules that are grouped together (e.g. multiple nodules)
                    - Step 2. for each nodule, extract the most detailed location, size, composition, echogenicity, shape, margin, echogenic foci, TI-RADS (these are features) that can be directly found and not inferred from the report. 
                    If the information of the feature does not exist in the radiology report, report "None" as the value. If the report directly mentions that there is no calcification, report echogenic foci as "no calcification".
                    - Step 3. change size to mm if written in cm. If size has more than one dimensions (e.g. 33 x 21 x 42 mm), report the largest value of all dimensions (e.g. 42mm).
                    - Step 4. Check if the the extracted information actually appear in the radiology report and assigned to correct nodule and features.
                
                3. **Extract Biopsy Report details (follow each step):**
                    - Step 1. Idnetify all samples biopsied in the biopsy report (under "FINAL DIAGNOSIS" section) and categorize them either as thyroid nodule and lymph node (e.g. level).
                    - Step 2. For each biopsied thyroid nodule, identify the location and size of the nodule under "Final Diagnosis" section.
                    - Step 3. For each biopsied thyroid nodule, check if more detailed location and size is available in the rest of the biopsy report.
                    - Step 4. If multiple detailed description from Step3 can be matched to nodule description from Step2, report the size and location ffrom Step 2. 
                    - Step 5. If there is only one detailed description from Step 3 can be matched to nodule description from Step 2 report the size and location from Step 3.
                    - Step 6. For each biopsied lymph node, report its biopsy result and level information only if they appear in the biopsy report.
                    - Step 7. Change size to mm if written in cm. If size has more than one dimensions (e.g. 33 x 21 x 42 mm), report the largest value of all dimensions (e.g. 42mm).
                    - Step 8. If the biopsy result is benign, report as simply "benign" but report the full description when malignent.

                4. **Identify Nodule Matching:**
                    - Step 1. For each biopsied thyroid nodule (not lymph node), check all nodules in the radiology report that can be matched based on size and location from previous steps.
                    - Step 2. For each biopsied thyroid nodule (not lymph node), if there are multiple possible matches or if it has no possible match, report the matched radiology report nodule as "Unknown" and confidence "Low".
                    - Step 3. For each biopsied thyroid noduel (not lymph node), if there is only one possible match, report the matched radiology report nodule and confidence "high"
                
                5. **Structure the Output in YAML:**
                - The final output should be a YAML document with a top-level keys are "radiology_report_nodules", "biopsy_report_nodules", "biopsy_report_lymph_nodes". "nodule_matching"
                - Use exactly the field names and structure as shown below.
                
                **YAML Structure:**
                    radiology_report_nodules:
                    -   nodule_id: id as integer
                        location: location information
                        size: size in mm
                        composition: [""]
                        echogenicity: 
                        shape: ['wider-than-tall', "taller-than-wide"]
                        margin: ["smooth", "ill-defined", "lobulated", "irregular", "extrathyroidal extension", "other", "Unknown"]
                        echogenic_foci: echogenic foci or "Unknow" if not available
                        TI-RADS_Score: integer
                
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
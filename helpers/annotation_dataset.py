import re

def parse_size(size_str):
    # Normalize string
    s = size_str.lower().strip()
    
    # If size is unknown or not specified, return "Unknown"
    if s in ["", "unknown", "none", "not specified"]:
        return "Unknown"
    
    # Determine unit
    if "mm" in s:
        unit = "mm"
    elif "cm" in s:
        unit = "cm"
    else:
        raise ValueError("No valid unit (mm or cm) found in input.")
    
    # Remove the unit from the string
    s = re.sub(unit, "", s, flags=re.IGNORECASE).strip()
    
    # Regex to extract numbers
    number_pattern = r'\d+(?:\.\d+)?'
    
    # Check if size has multiple dimensions (using "x")
    if "x" in s:
        dims = s.split("x")
        # Extract numbers from each dimension
        numbers = [float(re.search(number_pattern, dim).group()) 
                   for dim in dims if re.search(number_pattern, dim)]
        if not numbers:
            raise ValueError("No valid size found in input.")
        max_dim = max(numbers)
    else:
        match = re.search(number_pattern, s)
        if match:
            max_dim = float(match.group())
        else:
            raise ValueError("Size not provided or not in correct format.")
    
    # Convert to mm if necessary
    if unit == "cm":
        max_dim *= 10

    return max_dim

class NoduleMatch:
    def __new__(cls, match: dict):
        bx_id = match.get("biopsy_nodule_id")
        rad_id = match.get("matched_radiology_nodule_id") or match.get("radiology_nodule_id") or "unknown"

        # If bx_id is missing or unknown:
        if bx_id is None or bx_id.lower() == "unknown":
            # If radiology ID is provided, raise an error.
            if rad_id.lower() != "unknown":
                raise ValueError(
                    f"Invalid nodule match: Biopsy Nodule: {bx_id}, Radiology Nodule: {rad_id}"
                )
            # Otherwise, return None so that no object is created.
            return None

        # Otherwise, create the instance normally.
        return super().__new__(cls)
    
    def __init__(self, match:dict):
        self.bx_id = match.get("biopsy_nodule_id", "unknown")
        self.rad_id = match.get("matched_radiology_nodule_id") or match.get("radiology_noduel_id") or "unknown"
        # Determine match based on the radiology nodule ID.
        if self.rad_id.lower() == "unknown" or self.rad_id.strip() == "":
            self.match = False
        else:
            self.match = True
        self.reasoning = match.get("reasoning", "")
        self.confidence = match.get("matching_confidence", "")
        self.llm = bool(self.confidence)
    
    def __str__(self):
        return (f"Biopsy Nodule ID: {self.bx_id}, Radiology Nodule ID: {self.rad_id}, "
                f"Match: {self.match}, Reasoning: {self.reasoning}, "
                f"Matching Confidence: {self.confidence}")

class MatchReport:
    def __init__(self, match_list: list, patient_id: str = None):
        self.patient_id = patient_id  # patient_id is optional
        self.matches = []
        self.errors = []  # Store errors but do not re-raise

        for match in match_list:
            try:
                self.matches.append(NoduleMatch(match))
            except Exception as e:
                self.errors.append(str(e))
                print(f"Invalid nodule (Patient ID: {self.patient_id}): {e}")

    def __str__(self):
        report_output = []

        # Print valid nodules
        if self.matches:
            report_output.append("**Valid Matches:**")
            report_output.extend(str(match) for match in self.matches)

        # Print errors
        if self.errors:
            report_output.append("\n**Invalid Matches:**")
            for error_msg in self.errors:
                if self.patient_id:
                    report_output.append(f"Patient ID: {self.patient_id}, Error: {error_msg}")
                else:
                    report_output.append(f"Error: {error_msg}")

        return "\n".join(report_output) if report_output else "No nodules in the report."

class MatchDataset:
    def __init__(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary with patient IDs as keys.")

        self.reports = {}  # {patient_id: MatchReport}
        self.errors = []   # Collects ALL errors
        self.mrns = []

        for patient_id, patient_data in data.items():
            if "nodule_matching" not in patient_data:
                self.errors.append((patient_id, "Missing 'nodule_matching' key"))
                continue  # Skip this patient but continue processing others
            self.mrns.append(patient_id)
            
            report = MatchReport(patient_data["nodule_matching"], patient_id)
            self.reports[patient_id] = report

            # Collect errors from the report and store them in RadDataset
            for error_msg in report.errors:
                self.errors.append((patient_id, error_msg))

    def __str__(self):
        report_output = []
        # Print all errors
        if self.errors:
            report_output.append("\n**All Errors:**")
            for patient_id, error_msg in self.errors:
                report_output.append(f"Patient ID: {patient_id}, {error_msg}")

        # Print valid patient reports
        if self.reports:
            report_output.append("**Valid Patient Reports:**")
            for patient_id, report in self.reports.items():
                report_output.append(f"\n**Patient ID: {patient_id}**\n{report}")

        return "\n".join(report_output) if report_output else "RadDataset contains no valid patient reports."

class RadNodule:
    def __new__(cls, nodule: dict):
        # Check if both nodule_id and location are missing
        if not nodule.get("nodule_id") and not nodule.get("location"):
            # Return None to indicate no object should be created
            return None
        # Otherwise, proceed with normal object creation
        return super().__new__(cls)
    
    def __init__(self, nodule: dict):
        self.id = None
        try:
            # Require an ID (raise error if missing)
            if not nodule.get("nodule_id"):
                raise ValueError("Nodule does not have an ID")
            self.id = int(nodule["nodule_id"])

            # Validate location; raise error if missing.
            if not nodule.get("location"):
                raise ValueError("Location is empty")
            self.location = nodule["location"]

            # Validate size (convert from mm)
            size_str = str(nodule.get("size")).strip().lower()
            words_to_remove = ["up to", "sub-", "approximately"]
            pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
            size_str = re.sub(pattern, '', size_str.lower()).strip()

            # Identify unit and strip from text
            try:
                self.size = parse_size(nodule["size"])
            except ValueError as e:
                raise ValueError(f"Rad Nodule ID {self.id}: {str(e)} Input: {nodule['size']}")

            # Store optional attributes
            self.composition = nodule.get("composition", "Unknown")
            self.echogenicity = nodule.get("echogenicity", "Unknown")
            self.shape = nodule.get("shape", "Unknown")
            self.margin = nodule.get("margin", "Unknown")
            self.echogenic_foci = nodule.get("echogenic_foci", "Unknown")
            self.TIRADS = nodule.get("TI-RADS_Score", "Unknown")

        except Exception as e:
            raise ValueError(f"{str(e)}")

    def __str__(self):
        if self.id:
            return (f"Nodule ID: {self.id}, Location: {self.location}, Size: {self.size}mm, "
                    f"Composition: {self.composition}, Echogenicity: {self.echogenicity}, "
                    f"Shape: {self.shape}, Margin: {self.margin}, "
                    f"Echogenic Foci: {self.echogenic_foci}, TI-RADS Score: {self.TIRADS}")
        else:
            return "Empty nodule"

class RadReport:
    def __init__(self, nodule_list: list, patient_id: str = None):
        self.patient_id = patient_id  # patient_id is optional
        self.nodules = []
        self.errors = []  # Store errors but do not re-raise

        for nodule in nodule_list:
            try:
                nodule_obj = RadNodule(nodule)
                if nodule_obj is None:
                    continue
                self.nodules.append(nodule_obj)
            except Exception as e:
                nodule_id = nodule.get("nodule_id", "Unknown")
                self.errors.append((nodule_id, str(e)))
                print(f"Invalid nodule (Patient ID: {self.patient_id}, Nodule ID: {nodule_id}): {e}")
        
        self.locations = []
        for nodule in self.nodules:
            self.locations.append(nodule.location)

    def __str__(self):
        report_output = []

        # Print valid nodules
        if self.nodules:
            report_output.append("**Valid Nodules:**")
            report_output.extend(str(nodule) for nodule in self.nodules)

        # Print errors
        if self.errors:
            report_output.append("\n**Invalid Nodules:**")
            for nodule_id, error_msg in self.errors:
                if self.patient_id:
                    report_output.append(f"Patient ID: {self.patient_id}, Nodule ID: {nodule_id}, Error: {error_msg}")
                else:
                    report_output.append(f"Nodule ID: {nodule_id}, Error: {error_msg}")

        return "\n".join(report_output) if report_output else "No nodules in the report."

class RadDataset:
    def __init__(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary with patient IDs as keys.")

        self.reports = {}  # {patient_id: RadReport}
        self.errors = []   # Collects ALL errors
        self.mrns = []

        for patient_id, patient_data in data.items():
            if "radiology_report_nodules" not in patient_data:
                self.errors.append((patient_id, "Missing 'radiology_report_nodules' key"))
                continue  # Skip this patient but continue processing others
            self.mrns.append(patient_id)
            
            report = RadReport(patient_data["radiology_report_nodules"], patient_id)
            self.reports[patient_id] = report

            # Collect errors from the report and store them in RadDataset
            for nodule_id, error_msg in report.errors:
                self.errors.append((patient_id, f"Nodule ID {nodule_id}: {error_msg}"))

    def __str__(self):
        report_output = []
        # Print all errors
        if self.errors:
            report_output.append("\n**All Errors:**")
            for patient_id, error_msg in self.errors:
                report_output.append(f"Patient ID: {patient_id}, {error_msg}")

        # Print valid patient reports
        if self.reports:
            report_output.append("**Valid Patient Reports:**")
            for patient_id, report in self.reports.items():
                report_output.append(f"\n**Patient ID: {patient_id}**\n{report}")

        return "\n".join(report_output) if report_output else "RadDataset contains no valid patient reports."

class BxNodule:
    def __init__(self, nodule:dict):
        self.id = None
        try:
            # Try getting nodule_id
            if not nodule.get("nodule_id"):
                if not nodule.get("biopsy_result") and not nodule.get("location"):
                    return None
                else:
                    raise ValueError("Nodule does not have an ID")
                
            try:
                self.id = int(nodule["nodule_id"])
            except:
                self.id = str(nodule["nodule_id"])

            # try getting location
            self.location = nodule.get("location", "Unknown")

            # try getting size
            try:
                self.size = parse_size(nodule["size"])
            except ValueError as e:
                raise ValueError(f"Rad Nodule ID {self.id}: {str(e)} Input: {nodule['size']}")

            # try getting biopsy result
            if not nodule.get("biopsy_result"):
                self.result = "Unknown"
                raise ValueError("Nodule does not have biopsy result")
            self.result = nodule.get("biopsy_result")
            
        except Exception as e:
            raise ValueError(str(e))
    
    def __str__(self):
        if self.id:
            return (f"Nodule ID: {self.id}, Location: {self.location}, Size(in mm): {self.size}, "
                    f"Biopsy Result: {self.result}")
        else:
            return "Empty nodule"

class BxReport:
    def __init__(self, nodule_list: list, patient_id: str = None):
        self.patient_id = patient_id  # patient_id is optional
        self.nodules = []
        self.errors = []  # Store errors but do not re-raise

        for nodule in nodule_list:
            try:
                nodule_obj = BxNodule(nodule)
                if nodule_obj:
                    self.nodules.append(nodule_obj)
            except Exception as e:
                nodule_id = nodule.get("nodule_id", "Unknown")
                self.errors.append((nodule_id, str(e)))
                print(f"Invalid nodule (Patient ID: {self.patient_id}, Nodule ID: {nodule_id}): {e}")

    def __len__(self):
        return len(self.nodules)
    
    def __str__(self):
        report_output = []

        # Print valid nodules
        if self.nodules:
            report_output.append("**Valid Nodules:**")
            report_output.extend(str(nodule) for nodule in self.nodules)

        # Print errors
        if self.errors:
            report_output.append("\n**Invalid Nodules:**")
            for nodule_id, error_msg in self.errors:
                if self.patient_id:
                    report_output.append(f"Patient ID: {self.patient_id}, Nodule ID: {nodule_id}, Error: {error_msg}")
                else:
                    report_output.append(f"Nodule ID: {nodule_id}, Error: {error_msg}")

        return "\n".join(report_output) if report_output else "No nodules in the report."

class BxDataset:
    def __init__(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary with patient IDs as keys.")

        self.reports = {}  # {patient_id: BxReport}
        self.errors = []   # Collects ALL errors
        self.mrns = []

        for patient_id, patient_data in data.items():
            if "biopsy_report_nodules" not in patient_data:
                self.errors.append((patient_id, "Missing 'biopsy_report_nodules' key"))
                continue  # Skip this patient but continue processing others
            self.mrns.append(patient_id)
            
            report = BxReport(patient_data["biopsy_report_nodules"], patient_id)
            self.reports[patient_id] = report

            # Collect errors from the report and store them in RadDataset
            for nodule_id, error_msg in report.errors:
                self.errors.append((patient_id, f"Nodule ID {nodule_id}: {error_msg}"))

    def __str__(self):
        report_output = []
        # Print all errors
        if self.errors:
            report_output.append("\n**All Errors:**")
            for patient_id, error_msg in self.errors:
                report_output.append(f"Patient ID: {patient_id}, {error_msg}")

        # Print valid patient reports
        if self.reports:
            report_output.append("**Valid Patient Reports:**")
            for patient_id, report in self.reports.items():
                report_output.append(f"\n**Patient ID: {patient_id}**\n{report}")

        return "\n".join(report_output) if report_output else "BxDataset contains no valid patient reports."

class BxLymphNode:
    def __new__(cls, lymph_node: dict):
        if not lymph_node:
            return None
        node_id = lymph_node.get("lymph_node_id")
        if not node_id:
            return None
        return super().__new__(cls)

    def __init__(self, lymph_node: dict):
        self.id = lymph_node.get("lymph_node_id")
        self.level = lymph_node.get("level", "Unknown")
        self.result = lymph_node.get("biopsy_result")
        if not self.result:
            raise ValueError(f"Missing Biopsy result for Lymph Node {self.id}")
        
    def __str__(self):
        # If this object was successfully created, we assume id and result are valid.
        return f"Lymph Node ID: {self.id}, Level: {self.level}, Biopsy Result: {self.result}"

class LNReport:
    def __init__(self, LN_list:list, patient_id: str = None):
        self.patient_id = patient_id
        self.LN = []
        self.errors = []
        if LN_list:
            for ln in LN_list:
                LymphNode = BxLymphNode(ln)
                if LymphNode is None:
                    continue
                else:
                    try:
                        self.LN.append(LymphNode)
                    except Exception as e:
                        self.errors.append((LymphNode.id, str(e)))
                        print(f"Invalid nodule (Patient ID: {self.patient_id}, Nodule ID: {LymphNode.id}): {e}")
        
    def __len__(self):
        return len(self.LN)
    
    def __str__(self):
        report_output = []

        # Print valid nodules
        if self.LN:
            report_output.append("**Valid Lymph Nodes:**")
            report_output.extend(str(ln) for ln in self.LN)

        # Print errors
        if self.errors:
            report_output.append("\n**Invalid Nodules:**")
            for LN_id, error_msg in self.errors:
                if self.patient_id:
                    report_output.append(f"Patient ID: {self.patient_id}, Lymph Node ID: {LN_id}, Error: {error_msg}")
                else:
                    report_output.append(f"Lymph Node ID: {LN_id}, Error: {error_msg}")

        return "\n".join(report_output) if report_output else "No Lymph Node in the report."

class LNDataset:
    def __init__(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary with patient IDs as keys.")

        self.reports = {}  # {patient_id: LNReport}
        self.errors = []   # Collects ALL errors
        self.mrns = []

        for patient_id, patient_data in data.items():
            lymph_node_data = patient_data.get("biopsy_report_lymph_nodes")
            if not lymph_node_data:
                continue
            self.mrns.append(patient_id)
            report = LNReport(lymph_node_data, patient_id)
            self.reports[patient_id] = report

            # Collect errors from the report and store them in RadDataset
            for nodule_id, error_msg in report.errors:
                self.errors.append((patient_id, f"Nodule ID {nodule_id}: {error_msg}"))

    def __str__(self):
        report_output = []
        # Print all errors
        if self.errors:
            report_output.append("\n**All Errors:**")
            for patient_id, error_msg in self.errors:
                report_output.append(f"Patient ID: {patient_id}, {error_msg}")

        # Print valid patient reports
        if self.reports:
            report_output.append("**Valid Patient Reports:**")
            for patient_id, report in self.reports.items():
                report_output.append(f"\n**Patient ID: {patient_id}**\n{report}")

        return "\n".join(report_output) if report_output else "BxDataset contains no valid patient reports."

class RadPathDataset:
    def __init__(self, data: dict):
        """
        Combines radiology, biopsy, lymph node, and matching datasets.
        
        The input `data` should be a dictionary with patient IDs as keys.
        It may contain the following keys for each patient (if available):
          - "radiology_report_nodules"
          - "biopsy_report_nodules"
          - "biopsy_report_lymph_nodes"
          - "nodule_matching"
        """
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary with patient IDs as keys.")

        # Create each dataset from the same data dictionary.
        self.rad = RadDataset(data)
        self.bx = BxDataset(data)
        self.ln = LNDataset(data)
        self.match = MatchDataset(data)
        
        # Combine MRNs (patient IDs) from all datasets.
        self.mrns = set(self.rad.mrns) | set(self.bx.mrns) | \
                    set(self.ln.mrns) | set(self.match.mrns)
        
        # Combine errors from all datasets.
        # Each error is stored as a tuple: (source, patient_id, error_message)
        self.errors = []
        self.errors.extend([("RadDataset", patient_id, err) 
                            for patient_id, err in self.rad.errors])
        self.errors.extend([("BxDataset", patient_id, err) 
                            for patient_id, err in self.bx.errors])
        self.errors.extend([("LNDataset", patient_id, err) 
                            for patient_id, err in self.ln.errors])
        self.errors.extend([("MatchDataset", patient_id, err) 
                            for patient_id, err in self.match.errors])
    
    def __str__(self):
        output = []
        output.append("=== Radiology Reports ===")
        output.append(str(self.rad))
        
        output.append("\n=== Biopsy Reports ===")
        output.append(str(self.bx))
        
        output.append("\n=== Lymph Node Reports ===")
        output.append(str(self.ln))
        
        output.append("\n=== Nodule Matching Reports ===")
        output.append(str(self.match))
        
        if self.errors:
            output.append("\n=== Combined Errors ===")
            for source, patient_id, error in self.errors:
                output.append(f"Source: {source}, Patient ID: {patient_id}, Error: {error}")
        
        return "\n".join(output)
    

import re
from helpers.annotation_dataset import *

def parse_location(location: str) -> tuple[list[str], list[str]]:
    """
    Parse location string into primary and secondary locations
    """
    words_to_remove = ["thyroid", "gland", "nodule", "pole", "to", "of", "the", ",", "adjacent", "exophytic", "in", "lobe"]
    pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
    cleaned_location = re.sub(pattern, '', location.lower())
    cleaned_location = re.sub(r'[^\w\s/]', '', cleaned_location)  
    location_list = cleaned_location.strip().replace('/', ' ').split()

    # Find words starting with "isth"
    isthmus_indices = [i for i, word in enumerate(location_list) if word.startswith("isth")]

    primary_location = []
    secondary_location = location_list.copy()

    if isthmus_indices:
        primary_location.append("isthmus")

        # Check if "left" or "right" is NOT directly before an "isthmus" word
        for side in ["left", "right"]:
            if side in location_list:
                side_index = location_list.index(side)
                if not any(side_index + 1 == i for i in isthmus_indices):  # Ensure it's not next to "isthmus"
                    primary_location.append(side)

        # Remove primary locations from secondary
        secondary_location = [word for i, word in enumerate(location_list) if word not in primary_location and not word.startswith("isth")]

    else:
        # If no "isthmus", assign primary as "left" or "right" (whichever comes first)
        for side in ["left", "right"]:
            if side in location_list:
                primary_location.append(side)
                break  # Ensure we only take the first one found

        # Remove primary locations from secondary
        secondary_location = [word for word in location_list if word not in primary_location]

    return primary_location, secondary_location

class BaseAccuracy:
    def __init__(self):
        self.correct = 0
        self.partial_correct = 0
        self.incorrect = 0
        self.correct_info = {}
        self.partial_correct_info = {}
        self.incorrect_info = {}
        
    def __call__(self, mrn, manual, llm):
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if manual == llm:
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy
    
    @property
    def accuracy(self):
        """Return the current accuracy as a float (0.0 if no comparisons have been made)."""
        total = self.correct + self.incorrect
        return 0.0 if total == 0 else self.correct / total
    def __str__(self):
        all_info = ["**Incorrect Locations**"]
        for mrn, info_list in self.incorrect_info.items():
            all_info.append(f"\nMRN: {mrn}")
            for i, info in enumerate(info_list):
                for key, value in info.items():
                    all_info.append(f"{key}_{i}: {value}")
                all_info.append("---")
        return "\n".join(all_info) if self.incorrect_info else "100% Accuracy"

class LocationAccuracy(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        # return the updated accuracy 
        manual_primary, manual_secondary = self.parse_location(manual)
        llm_primary, llm_secondary = self.parse_location(llm)
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if manual_primary == llm_primary and set(manual_secondary) == set(llm_secondary):
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy
    
    def parse_location(self, location: str) -> tuple[list[str], list[str]]:
        """
        Parse a location string into primary and secondary locations.

        Parameters:
            location (str): The location string to be parsed.

        Returns:
            tuple[list[str], list[str]]: A tuple containing two lists:
                - Primary location elements.
                - Secondary location elements.
        """
        if not location:  # Handle empty or None inputs gracefully.
            return [], []

        words_to_remove = ["deep", "mass", 'nodule,', 'sided', "bed", "thyroid", "gland", "nodule", "pole", "to", "of", "the", ",", "adjacent", "exophytic", "in", "lobe", "surface"]
        pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
        cleaned_location = re.sub(pattern, '', location.lower())
        cleaned_location = re.sub(r'[^\w\s/]', '', cleaned_location)
        cleaned_location = re.sub(r'\d+(?:\.\d+)?(?:\s*(mm|cm))?', '', cleaned_location)
        location_list = cleaned_location.strip().replace('/', ' ').split()

        # Identify indices of words starting with "isth"
        isthmus_indices = [i for i, word in enumerate(location_list) if word.startswith("isth")]

        primary_location = []
        secondary_location = location_list.copy()

        if isthmus_indices:
            primary_location.append("isthmus")
            # Check if "left" or "right" is NOT directly before an "isthmus" word
            for side in ["left", "right"]:
                if side in location_list:
                    side_index = location_list.index(side)
                    if not any(side_index + 1 == i for i in isthmus_indices):
                        primary_location.append(side)
            # Remove primary keywords from secondary locations
            secondary_location = [
                word for i, word in enumerate(location_list)
                if word not in primary_location and not word.startswith("isth")
            ]
        else:
            # If "isthmus" is not found, consider the first occurrence of "left" or "right"
            for side in ["left", "right"]:
                if side in location_list:
                    primary_location.append(side)
                    break  # Only the first one is taken as primary.
            secondary_location = [self.standardize_secondary(word) for word in location_list if word not in primary_location]

        return primary_location, secondary_location

    def standardize_secondary(self, secondary_loc_str: str):
        mapping = {
            "lower": "inferior",
            "upper": "superior",
            "middle": "mid"
        }
        return mapping.get(secondary_loc_str, secondary_loc_str)

class TIRADSAccucarcy(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        manual_tirads = self.parse_TIRADS(manual)
        llm_tirads = self.parse_TIRADS(llm)
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if manual_tirads == llm_tirads:
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy

    def parse_TIRADS(self, input):
        pattern = r'\b\d\b'
        if input:
            cleaned = re.sub(r'\([^)]*\)', '', input)
            matches = re.findall(pattern, cleaned)
            matches = [match for match in matches if int(match) < 5]
            if matches and len(matches) == 1:
                return [int(matches[0])]
            else:
                return matches
        else:
            return []

class FociAccuracy(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        manual_foci = self.clean(manual)
        llm_foci = self.clean(llm)
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if manual_foci == llm_foci:
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy
    
    def clean(self, text):
        words_to_remove = ["few", "a few",",","unknown", "with ", "\(rim\) ", "suggestive of colloid", "definite ", "additional ", "multiple ", "likely "]
        pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
        cleaned_text = re.sub(pattern, '', text.lower())
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s_]', '', cleaned_text)
        cleaned_text = re.sub(r'without\s+associated\s+calcifications', 'no calcifications', cleaned_text)
        cleaned_text = re.sub(r'no\s+associated\s+calcifications', 'no calcifications', cleaned_text)
        cleaned_text = re.sub("associated ", "", cleaned_text)
        cleaned_text = re.sub(r'without\s+calcifications', 'no calcifications', cleaned_text)
        cleaned_text = re.sub(r'\bnone\b', 'no calcifications', cleaned_text)
        
        # Normalize calcification spelling
        cleaned_text = re.sub(r'calcification', 'calcifications', cleaned_text)
        
        # Final cleanup: remove extra whitespace
        cleaned_text = cleaned_text.strip()

        return cleaned_text

class FociAccuracy2(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        manual_foci = self.clean(manual)
        llm_foci = self.clean(llm)
        manual_foci_set = set(manual_foci.split())
        llm_foci_set = set(llm_foci.split())

        if manual_foci == llm_foci:
            self.correct += 1
            self.correct_info[mrn] = {"Manual Annotation": manual, "LLM Annotation": llm}
        else:
            self.incorrect += 1
            self.incorrect_info[mrn] = {"Manual Annotation": manual, "LLM Annotation": llm}
        return self.accuracy
    
    def clean(self, text):
        #words_to_remove = ["few", "a few",",","unknown", "with ", "\(rim\) ", "suggestive of colloid", "definite ", "additional ", "multiple ", "likely "]
        #pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
        #cleaned_text = re.sub(pattern, '', text.lower())
        cleaned_text = text.lower()
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s_]', '', cleaned_text)
        cleaned_text = re.sub(r'without\s+associated\s+calcifications', 'no calcifications', cleaned_text)
        cleaned_text = re.sub(r'no\s+associated\s+calcifications', 'no calcifications', cleaned_text)
        cleaned_text = re.sub("associated ", "", cleaned_text)
        cleaned_text = re.sub(r'without\s+calcifications', 'no calcifications', cleaned_text)
        cleaned_text = re.sub(r'\bnone\b', 'no calcifications', cleaned_text)
        
        # Normalize calcification spelling
        cleaned_text = re.sub(r'calcification', 'calcifications', cleaned_text)
        
        # Final cleanup: remove extra whitespace
        cleaned_text = cleaned_text.strip()

        return cleaned_text
    
class MarginAccuracy(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        manual_margin = self.clean(manual)
        llm_margin = self.clean(llm)
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if manual_margin == llm_margin:
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy
    
    def clean(self, text):
        words_to_remove = ["suspicious for", "borders", "poorly defined", "somewhat", "relatively", "margins", "margin"]
        pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
        cleaned_text = re.sub(pattern, '', text.lower()) # remove words in the list words_to_remove
        cleaned_text = re.sub(r'\-', ' ', cleaned_text) # remove "-" e.g. ill-defined
        cleaned_text = re.sub(r'\([^)]*\)', '', cleaned_text) # remove TI-RADS score within parnethesis
        cleaned_text = re.sub(r'none', '', cleaned_text) # change "none" to empty string
        cleaned_text = re.sub(r'unknown', '', cleaned_text) # change "unknown" to empty string
        cleaned_text = cleaned_text.strip()

        return cleaned_text
    
class ShapeAccuracy(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        manual_shape = self.clean(manual)
        llm_shape = self.clean(llm)
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if manual_shape == llm_shape:
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy
    
    def clean(self, text):
        words_to_remove = ["margins"] # copied from margin loss and does nothing for the shape (just a place holder for future words to remove)
        pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
        cleaned_text = re.sub(pattern, '', text.lower()) # remove words in the list words_to_remove
        cleaned_text = re.sub(r'\([^)]*\)', '', cleaned_text) # remove TI-RADS score within parnethesis
        cleaned_text = re.sub(r'\-', ' ', cleaned_text) # remove "-" e.g. wider-than-tall
        cleaned_text = re.sub(r'none', '', cleaned_text) # change "none" to empty string
        cleaned_text = re.sub(r'unknown', '', cleaned_text) # change "unknown" to empty string


        cleaned_text = cleaned_text.strip()

        return cleaned_text

class EchoAccuracy(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        manual_echo = self.clean(manual)
        llm_echo = self.clean(llm)
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if manual_echo == llm_echo:
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy
    
    def clean(self, text):
        words_to_remove = ["margins", "cannot be determined", "none", "unknown" ]
        pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
        cleaned_text = re.sub(pattern, '', text.lower())
        cleaned_text = re.sub(r'hypochoic', 'hypoechoic', cleaned_text)
        cleaned_text = re.sub(r'hyperchoic', 'hyperechoic', cleaned_text)
        cleaned_text = re.sub(r'none', '', cleaned_text)
        cleaned_text = re.sub(r'unknown', '', cleaned_text)

        cleaned_text = cleaned_text.strip()

        return cleaned_text
    
class CompAccuracy(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        manual_comp = self.clean(manual)
        llm_comp = self.clean(llm)
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if manual_comp == llm_comp:
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy
    
    def map_compositon(text):
        comp_map = {"solid": [], "Mixed cystic and solid": [], "Spongiform": [], "Cystic": []}
        solid_list = [r"solid or almost completely (>95% solid)", "solid", "predominantly solid",
                      "complex but predominantly solid", ]
        mixed_list = ["mixed solid/cystic", "heterogeneous", "mixed cystic and solid"]

    def clean(self, text):
        #words_to_remove = ["with", "nodule", "and ", "complex", "but ", "unknown", "cannot be determined due to calcifications"]
        words_to_remove = ["with", "nodule ", "nodule", "and ", "complex", "but ", "unknown", "cannot be determined due to calcifications"]
        pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
        cleaned_text = re.sub(pattern, '', text.lower())
        cleaned_text = re.sub(r'[^\w\s/]', '', cleaned_text)
        cleaned_text = re.sub(r'none', '', cleaned_text)
        cleaned_text = re.sub(r'/', ' ', cleaned_text)
        cleaned_text = cleaned_text.replace("heterogeneous", "mixed cystic and solid").strip()

        return cleaned_text

class BxResultAccuracy(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        manual_primary, manual_secondary = self.parse_result(manual)
        llm_primary, llm_secondary = self.parse_result(llm)
        # Determine if the annotation is correct based on the primary (and secondary, if applicable) labels.
        is_correct = (manual_primary == llm_primary and 
                      (manual_primary == "benign" or set(manual_secondary) <= set(llm_secondary)))
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if is_correct:
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy
    
    def parse_result(self, text):
        text_list = text.lower().replace(",", "").replace("- ", "").split()
        if "benign" in text_list or "hashimoto's" in text_list or "hashimoto" in text_list:
            primary = "benign"
            secondary = None
        else:
            primary = "malignant"
            secondary = text.lower().replace(",", "").replace("- ", "").replace("malignant", "").replace(".","").replace("thyroid", "").split()
        return primary, secondary

class LNLevelAccuracy(BaseAccuracy):
    def __call__(self, mrn, manual, llm):
        manual_comp = self.clean(manual)
        llm_comp = self.clean(llm)
        entry = {"Manual Annotation": manual, "LLM Annotation": llm}
        if manual_comp == llm_comp:
            self.correct += 1
            self.correct_info.setdefault(mrn, []).append(entry)
        else:
            self.incorrect += 1
            self.incorrect_info.setdefault(mrn, []).append(entry)
        return self.accuracy
    
    def clean(self, text):
        words_to_remove = ["lymph", "node", "level", "right", "left"]
        pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'        
        cleaned_text = re.sub(pattern, '', text.lower())
        cleaned_text = self.roman_to_arabic(cleaned_text.strip())
        return cleaned_text
    
    def roman_to_arabic(self, input_str: str) -> int:
        input_str = input_str.upper().strip()
        mapping = {
            "I": '1',
            "II": '2',
            "III": '3',
            "IV": '4',
            "V": '5',
            "VI": '6',
            "VII": '7',
            "VIII": '8',
            "IX": '9',
        }
        if input_str not in list(mapping.keys()):
            return input_str
        return mapping[input_str]

class MatchAccuracy(BaseAccuracy):
    def __init__(self):
        super().__init__()
        self.TP = {}
        self.TN = {}
        self.FP = {}
        self.FN = {}

    def __call__(self, mrn, manual: NoduleMatch, llm: NoduleMatch):
        manual_bxid = self.clean(manual.bx_id)
        manual_radid = self.clean(manual.rad_id)
        llm_bxid = self.clean(llm.bx_id)
        llm_radid = self.clean(llm.rad_id)

        # Calculate accuracy
        if manual_bxid == llm_bxid and manual_radid == llm_radid:
            self.correct += 1
            self.correct_info[f"{mrn}_{manual.bx_id}"] = {"Manual Annotation": (manual.bx_id, manual.rad_id), 
                                      "LLM Annotation": (llm.bx_id, llm.rad_id), "Confidence": llm.confidence}
        else:
            self.incorrect += 1
            self.incorrect_info[f"{mrn}_{manual.bx_id}"] = {"Manual Annotation": (manual.bx_id, manual.rad_id), 
                                      "LLM Annotation": (llm.bx_id, llm.rad_id), "Confidence": llm.confidence}  
    
        self.calculate_confusion_matrix(mrn, manual, llm)

        return self.accuracy
    
    def __str__(self):
        all_info = ["**Incorrect Matches**",]
        all_info.append(f"\n{self.incorrect_info.keys()}")
        for mrn, info in self.incorrect_info.items():
            all_info.append(f"\n{mrn}")
            for key, value in info.items():
                all_info.append(f"{key}: {value}")
        all_info.append("\n**Correct Matches with 'Low' Confidence**")
        n = 0
        for mrn, info in self.correct_info.items():
            if info["Confidence"] == "Low":
                n+=1
                all_info.append(f"\n{mrn}")
                for key, value in info.items():
                    all_info.append(f"{key}: {value}")
        all_info.append(str(n))
        return "\n".join(all_info) if self.incorrect_info else "100% Accuracy"

    def calculate_confusion_matrix(self, mrn, manual: NoduleMatch, llm: NoduleMatch):
        match_manual = manual.match
        match_llm = llm.match
        if "or" in manual.rad_id:
            match_manual = False
        if "or" in llm.rad_id:
            match_llm = False
        if match_manual and match_llm:
            self.TP[f"{mrn}_{manual.bx_id}"] = {"Manual Annotation": (manual.bx_id, manual.rad_id), 
                            "LLM Annotation": (llm.bx_id, llm.rad_id)}
        elif not match_manual and not match_llm:
            self.TN[f"{mrn}_{manual.bx_id}"] = {"Manual Annotation": (manual.bx_id, manual.rad_id), 
                            "LLM Annotation": (llm.bx_id, llm.rad_id)}
        elif not match_manual and match_llm:
            self.FP[f"{mrn}_{manual.bx_id}"] = {"Manual Annotation": (manual.bx_id, manual.rad_id), 
                            "LLM Annotation": (llm.bx_id, llm.rad_id)}
        elif match_manual and not match_llm:
            self.FN[f"{mrn}_{manual.bx_id}"] = {"Manual Annotation": (manual.bx_id, manual.rad_id), 
                            "LLM Annotation": (llm.bx_id, llm.rad_id)}

    def clean(self, text):
        text = str(text)
        pattern = r'\b\d'
        cleaned_text = re.sub(r'a', '1', text.lower())
        cleaned_text = re.sub(r'b', '2', cleaned_text)
        cleaned_text = re.sub(r'c', '3', cleaned_text)
        cleaned_text = re.sub(r'^\D+', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        if not cleaned_text or "or" in cleaned_text:
            cleaned_text = 'unknown'
        return cleaned_text
    
    @property
    def confusion_matrix(self):
        return (len(self.TP), len(self.TN), len(self.FP), len(self.FN))
    

"""
PyTorch Dataset implementations for the LLM inference system.
"""
from typing import Dict, Any, List
from torch.utils.data import Dataset

class PairedReportDataset(Dataset):
    """Dataset for paired report data (radiology and biopsy reports)."""
    
    def __init__(self, radreports: dict, bxreports: dict, mrns: list):
        self.mrns = mrns
        self.radreports = {mrn: radreports[mrn] for mrn in mrns}
        self.bxreports = {mrn: bxreports[mrn] for mrn in mrns}
    
    def __len__(self):
        return len(self.mrns)
    
    def __getitem__(self, index):
        mrn = self.mrns[index]
        return mrn, self.radreports[mrn], self.bxreports[mrn]

class SingleReportDataset(Dataset):
    """Dataset for single report data (e.g., radiology only)."""
    
    def __init__(self, reports: dict, mrns: list):
        self.mrns = mrns
        self.reports = {mrn: reports[mrn] for mrn in mrns}
    
    def __len__(self):
        return len(self.mrns)
    
    def __getitem__(self, index):
        mrn = self.mrns[index]
        return mrn, self.reports[mrn]
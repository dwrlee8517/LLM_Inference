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

    def filter_ids(self, ids: list):
        """Return a new PairedReportDataset containing only the provided ids."""
        filtered_ids = [mrn for mrn in ids if mrn in self.radreports and mrn in self.bxreports]
        return PairedReportDataset(self.radreports, self.bxreports, filtered_ids)

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

    def filter_ids(self, ids: list):
        """Return a new SingleReportDataset containing only the provided ids."""
        filtered_ids = [mrn for mrn in ids if mrn in self.reports]
        return SingleReportDataset(self.reports, filtered_ids)
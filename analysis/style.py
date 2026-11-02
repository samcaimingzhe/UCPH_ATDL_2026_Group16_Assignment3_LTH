"""Shared matplotlib style: Times New Roman for all figure text."""
import matplotlib


def apply():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Times-like math
        "pdf.fonttype": 42,          # embed TrueType so PDFs keep the font
        "ps.fonttype": 42,
    })

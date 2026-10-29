"""Reading .mca files."""
import numpy as np


def _to_number(text):
    value = float(text)
    return int(value) if value.is_integer() else value


def import_from_mca(MCA_input):
    """
    Returns
    -------
    ROIs      : array of [ROI_low, ROI_high] rows
    chData    : array with the counts, the index is the channel number
    live_time : array with the LIVE_TIME repeated once per ROI
    """
    ROI_list = []
    Data_list = []
    ROI_start = False
    Data_start = False
    LIVE_TIME = 0

    with open(MCA_input, "r") as file:
        for line in file:
            line_str = line.strip()
            parts = line_str.split()
            if not parts:
                continue

            # Live time
            if parts[0] == "LIVE_TIME":
                LIVE_TIME = _to_number(parts[2])

            # Where the ROIs start
            if line_str == "<<ROI>>":
                ROI_start = True
                continue

            # Where the DATA starts (the ROIs end here)
            if line_str == "<<DATA>>":
                ROI_start = False
                Data_start = True
                continue

            # End of the DATA listing
            if line_str == "<<END>>":
                Data_start = False
                continue

            if ROI_start:
                ROI = list(map(int, parts))
                ROI_list.append([ROI[0], ROI[1]])

            if Data_start:
                Data_list.append(int(line_str))

    return np.array(ROI_list), np.array(Data_list), np.ones(len(ROI_list)) * LIVE_TIME


def sort_ROI(ROI_input):
    """Sorts the ROIs in ascending order of their lower edge."""
    sort_mask = ROI_input[:, 0].argsort()
    return ROI_input[sort_mask]

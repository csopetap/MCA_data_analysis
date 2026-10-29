"""
The single callable entry point: analyze_folder().

For every .mca file in a folder it
  1. reads the ROIs, the spectrum and the live time,
  2. expands every ROI and fits the peak (errors: statistical + systematic),
  3. converts the peak position to energy:  E = a + b * channel   (skipped if a == b == -1),
  4. matches the energy to the database lines of the chosen decay chain(s),
  5. calculates A*eta = N / (t * I),
and writes one CSV per file: <output_folder>/CSV_Outputs/Result_<file name>.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .fitting import expand_and_fit_all_peaks
from .matching import load_database, match_peaks
from .plotting import PlotConfig
from .reader import import_from_mca, sort_ROI

CSV_SUBFOLDER = "CSV_Outputs"

OUTPUT_COLUMNS = [
    "LIVE_TIME", "ROI_input",
    "Energy measured (keV)", "Energy (keV)", "Intensity", "Intensity error", "line_matched",
    "Net area", "Net area error",
    "FWHM (Ch)", "dFWHM (Ch)",
    "A*eta", "A*eta Error", "fit_success",
]

# Only written if include_debug_columns=True
DEBUG_COLUMNS = [
    "ROI_recommended", "Centroid", "Total Area", "Net area stat error", "Net area syst error",
    "Isotope", "Decay chain", "fail_reason",
]

_ENERGY_COLUMNS = ["Energy measured (keV)", "Energy (keV)", "Intensity", "Intensity error",
                   "Isotope", "Decay chain"]


def _calibration_enabled(calibration_parameters):
    """(-1, -1) or None means: no calibration."""
    if calibration_parameters is None:
        return False
    a, b = calibration_parameters
    return not (a == -1 and b == -1)


def _add_A_eta(df):
    """A*eta = N / (t * I), the intensity is in percent in the database."""
    N, dN, t = df["Net area"], df["Net area error"], df["LIVE_TIME"]
    I, dI = df["Intensity"] / 100, df["Intensity error"] / 100
    df["A*eta"] = N / (t * I)
    df["A*eta Error"] = np.sqrt((dN / (I * t)) ** 2 + (N * dI / (t * I ** 2)) ** 2)
    return df


def _analyze_file(mca_path, calibration_parameters, calibrate, db, tolerance_keV, interactive,
                  fit_kwargs):
    ROI_input, chData, live_time = import_from_mca(mca_path)
    if len(ROI_input) == 0:
        print(f"Warning: no ROIs found in {mca_path.name}, skipping it.")
        return None

    ROI_sorted = sort_ROI(ROI_input)
    df = expand_and_fit_all_peaks(chData, ROI_sorted, file_stem=mca_path.stem, **fit_kwargs)
    df.insert(0, "LIVE_TIME", live_time[0])

    if calibrate:
        a, b = calibration_parameters
        df["Energy measured (keV)"] = a + b * df["Centroid"]
        labels = [f"{mca_path.name}, ROI {roi}" for roi in df["ROI_input"]]
        matched = match_peaks(df["Energy measured (keV)"].to_numpy(), labels, db,
                              tolerance_keV, interactive)
        df = pd.concat([df, matched], axis=1)
    else:
        for col in _ENERGY_COLUMNS:
            df[col] = np.nan
        df["line_matched"] = False

    df = _add_A_eta(df)

    msg = f"{mca_path.name}: {len(df)} peaks, {int((~df['fit_success']).sum())} failed fits"
    if calibrate:
        msg += f", {int(df['line_matched'].sum())} database lines matched"
    print(msg)
    return df


def analyze_folder(input_folder, output_folder, calibration_parameters=(-1, -1),
                   decay_chains=None, database_path="Intensity_Database.csv", tolerance_keV=5.0,
                   plot_folder=None, second_der_signif_toler=2, first_der_signif_toler = 1.5, search_max_low=20,
                   search_max_high=20, wROI_window = 10, shift_ratio=0.3, plot_ROI_expand=True, plotFit=True,
                   plotChi2=True, plotSystError=True, plotLang="eng", interactive=True,
                   include_debug_columns=False, max_relative_error=1.0):
    """
    Analyses every .mca file in `input_folder`. For each file a CSV named
    Result_<file name>.csv is written into <output_folder>/CSV_Outputs/ (right after the file is
    done, so finished files are kept even if a later one is stopped).

    Parameters
    ----------
    input_folder, output_folder : folders (output_folder is created if needed)
    calibration_parameters : (a, b) with E = a + b * channel. (-1, -1) or None skips the
        calibration, the database matching and A*eta (those columns are NaN, line_matched False).
    decay_chains : decay chain name or list of names from the first database column,
        e.g. ["238U", "232Th"]. None uses the whole database.
    database_path : the database CSV (columns by position: chain, isotope, energy, intensity %, error)
    tolerance_keV : matching tolerance between the measured and the database energy
    plot_folder : where Plot_Outputs is created (default: inside output_folder).
        Subfolders: Fit_plots, ROI_plots, chi2_plots, syst_err_plots.
    second_der_signif_toler, first_der_signif_toler, search_max_low, search_max_high, wROI_window, shift_ratio : ROI expansion and
        systematic error settings
    plot_ROI_expand, plotFit, plotChi2, plotSystError : which plots to make
    plotLang : 'eng' or 'hun' (language of the fit plots)
    interactive : if several database lines are in tolerance, ask which one to use.
        If False the closest one is used (with a warning).
    max_relative_error : a peak whose net area error is larger than this fraction of its net area
        is counted as unsuccessful (default 1.0 = error bigger than the value itself; None disables).
    include_debug_columns : also write recommended ROI, centroid, total area, stat/syst errors,
        isotope, decay chain and the failure reason.

    Columns: fit_success is False for peaks whose fit failed; line_matched is True if a database
    line was assigned to the peak.

    Returns a dict {mca file name: DataFrame} with the same tables that were written to the CSVs.
    """
    input_folder, output_folder = Path(input_folder), Path(output_folder)
    if plotLang not in ("eng", "hun"):
        raise ValueError("plotLang must be 'eng' or 'hun'")

    mca_files = sorted(p for p in input_folder.iterdir() if p.suffix.lower() == ".mca")
    if not mca_files:
        raise FileNotFoundError(f"No .mca files found in {input_folder}")

    csv_dir = output_folder / CSV_SUBFOLDER
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_root = Path(plot_folder) if plot_folder else output_folder / "Plot_Outputs"
    plot_cfg = PlotConfig.create(plot_root, plot_ROI_expand, plotFit, plotChi2, plotSystError, plotLang)

    calibrate = _calibration_enabled(calibration_parameters)
    db = load_database(database_path, decay_chains) if calibrate else None
    if not calibrate:
        print("No calibration given: energies, intensities and A*eta will be NaN.")

    fit_kwargs = dict(second_der_signif_toler=second_der_signif_toler,
                      first_der_signif_toler=first_der_signif_toler,
                      search_max_low=search_max_low, search_max_high=search_max_high, wROI_window = wROI_window,
                      shift_ratio=shift_ratio, plot_cfg=plot_cfg,
                      max_relative_error=max_relative_error)
    columns = OUTPUT_COLUMNS + (DEBUG_COLUMNS if include_debug_columns else [])

    results = {}
    for mca_path in mca_files:
        print(f"\n=== {mca_path.name} ===")
        df = _analyze_file(mca_path, calibration_parameters, calibrate, db, tolerance_keV,
                           interactive, fit_kwargs)
        if df is None:
            continue
        df = df[columns]
        out_path = csv_dir / f"Result_{mca_path.stem}.csv"
        df.to_csv(out_path, index=False)
        print(f"Written: {out_path}")
        results[mca_path.name] = df

    return results

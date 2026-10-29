"""
All plotting. Figures are built with matplotlib's object-oriented Figure class (no pyplot),
so there is no global state, nothing piles up between plots and no GUI backend is needed.
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

from .models import peak_plus_bkg_function, peak_function, bkg_function

PLOT_SUBFOLDERS = {
    "fit": "Fit_plots",
    "roi": "ROI_plots",
    "chi2": "Chi2_plots",
    "syst": "Syst_err_plots",
}

_LABELS = {
    "eng": dict(spectrum="Measured spectrum", roi="Expanded ROI", xlabel="Channel number",
                ylabel="Counts", total="Total fitted function", peak="Fitted peak component",
                bkg="Fitted background component", legend_fs=6),
    "hun": dict(spectrum="Mért spektrum", roi="Kitolt ROI", xlabel="Csatornaszám",
                ylabel="Beütések", total="Teljes illesztett függvény",
                peak="Illesztett csúcs komponens", bkg="Illesztett háttér", legend_fs=10),
}

_HUN_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "font.size": 14,
}


@dataclass
class PlotConfig:
    """Which plots to make, in which language, and where they go."""
    roi_expand: bool = True
    fit: bool = False
    chi2: bool = False
    syst_err: bool = True
    lang: str = "eng"
    roi_dir: Path = Path("Plot_Outputs/ROI_plots")
    fit_dir: Path = Path("Plot_Outputs/Fit_plots")
    chi2_dir: Path = Path("Plot_Outputs/chi2_plots")
    syst_dir: Path = Path("Plot_Outputs/syst_err_plots")

    @classmethod
    def create(cls, plot_root, roi_expand, fit, chi2, syst_err, lang):
        """Creates Plot_Outputs and its four subfolders and returns the config."""
        plot_root = Path(plot_root)
        dirs = {key: plot_root / name for key, name in PLOT_SUBFOLDERS.items()}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return cls(roi_expand, fit, chi2, syst_err, lang,
                   dirs["roi"], dirs["fit"], dirs["chi2"], dirs["syst"])


def plot_fit(chData, roi, popt, path, lang="eng"):
    lab = _LABELS[lang]
    with mpl.rc_context(_HUN_RC if lang == "hun" else {}):
        fig = Figure()
        ax = fig.subplots()
        x = np.arange(len(chData))
        mask = (x < roi[1] + 10) & (roi[0] - 10 < x)
        x_fine = np.linspace(x[mask][0], x[mask][-1], 1000)

        ax.errorbar(x[mask], chData[mask], yerr=np.sqrt(chData[mask]), linestyle='', marker='o',
                    markersize=3, capsize=2, color='C0', label=lab["spectrum"])
        ax.vlines(roi, colors='red', ymin=0, ymax=np.max(chData[mask]), label=lab["roi"])
        ax.plot(x_fine, peak_function(x_fine, *popt), label=lab["peak"])
        ax.plot(x_fine, bkg_function(x_fine, *popt), label=lab["bkg"])
        ax.plot(x_fine, peak_plus_bkg_function(x_fine, *popt), label=lab["total"], color = "black")
        ax.set_xlabel(lab["xlabel"])
        ax.set_ylabel(lab["ylabel"])
        ax.set_ylim(bottom=0)
        ax.grid()
        ax.legend(fontsize=lab["legend_fs"])
        fig.tight_layout()
        fig.savefig(path)


def plot_chi2(chData, roi, popt, path):
    x_all = np.arange(len(chData), dtype=float)
    mask = (x_all < roi[1]) & (roi[0] < x_all)
    # Only the channels inside the ROI are evaluated (the model overflows far from the peak)
    x = x_all[mask]
    counts = chData[mask]
    # Same uncertainty as in the fit (at least 1 count), so empty channels don't give inf
    sigma = np.max([np.sqrt(counts), np.ones(len(counts))], axis=0)
    residuals = (counts - peak_plus_bkg_function(x, *popt)) / sigma
    chi2 = np.sum(residuals ** 2)
    ndf = len(x) - 8

    fig = Figure()
    ax = fig.subplots()
    ax.errorbar(x, residuals, np.ones(len(x)), marker='o', color='black',
                linestyle='', capsize=3)
    ax.vlines(roi, colors='red', ymin=np.min(residuals - 1), ymax=np.max(residuals + 1),
              label="Expanded ROI", linestyles='--')
    ax.axhline(0, color='green')
    ax.set_xlabel('Channel number')
    ax.set_ylabel('Residual')
    ax.set_title(rf'$\chi^2 / \mathrm{{ndf}} = {chi2:.2f} / {ndf}$')
    ax.grid()
    ax.legend(fontsize=6)
    fig.savefig(path)


def plot_roi_expansion(chData, input_ROI, rec_ROI, diag, path):
    S_list = diag["S_list"]
    significance = diag["significance"]
    w_plot = 10

    fig = Figure()
    ax1 = fig.subplots()
    x = np.arange(len(chData))
    mask = (x < rec_ROI[1] + w_plot) & (rec_ROI[0] - w_plot < x)
    ymax = np.max(chData[mask])

    ax1.errorbar(x[mask], chData[mask], yerr=np.sqrt(chData[mask]), linestyle='', marker='o',
                 markersize=3, capsize=2, color='C0', label="Measured spectrum")
    ax1.vlines(input_ROI, colors='red', ymin=0, ymax=ymax, label="Input ROI")
    ax1.vlines(rec_ROI, colors='green', ymin=0, ymax=ymax, linestyles="dashed", label="Recommended ROI")
    ax1.set_xlabel('Channel number')
    ax1.set_ylabel('Counts', color='C0')
    ax1.tick_params(axis='y', color='C0')
    ax1.set_yscale('log')
    ax1.legend(fontsize=6, loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Second derivative significance', color='C1')
    delta_S = significance - S_list
    ok = mask & np.isfinite(delta_S)      # the padded end channels are inf
    ax2.errorbar(x[ok], delta_S[ok], linestyle='', marker='o', markersize=3, capsize=2,
                 color='C1', label=r'$\Delta S$')
    ax2.tick_params(axis='y', color='C1')
    ax2.set_ylim(0, 5)
    ax2.grid()
    ax2.legend(fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(path)


def plot_syst_error(areas, sigma_syst, n_low, n_high, step_low, step_high, path):
    if len(areas) < 2 or not np.isfinite(areas[0]) or areas[0] == 0:
        return
    A0 = areas[0]
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    ax.hist(areas / A0, label=f'sigma = {sigma_syst / A0 * 100:.2f}%')
    ax.set_title(f'n_low = {n_low}, step_low = {step_low}, n_high = {n_high}, step_high = {step_high}')
    ax.set_xlabel('Area/Area_0')
    ax.grid()
    ax.legend()
    fig.savefig(path)

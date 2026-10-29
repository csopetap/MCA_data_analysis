# MCA data analysis

`MCA_data_analysis` is a Python package for analyzing multichannel analyzer
(`.mca`) spectra from HPGe and NaI detectors. It reads spectrum data and
regions of interest (ROIs), expands and fits each ROI, optionally calibrates
channel numbers to energy, matches peaks to a gamma-ray database, and writes
CSV results and diagnostic plots.

The main entry point is:

```python
from mca_pipeline import analyze_folder
```

## Features

- Reads `.mca` files containing `LIVE_TIME`, `<<ROI>>`, `<<DATA>>`, and
  `<<END>>` sections.
- Sorts and expands each input ROI to estimate a suitable fitting window.
- Fits peaks with a Gaussian plus low-energy tail, step background, and linear
  background.
- Propagates statistical and ROI-variation systematic uncertainties.
- Converts fitted centroids to energy with the linear calibration
  `E = a + b * channel`.
- Matches calibrated peaks to selected decay-chain lines in the included CSV
  database.
- Calculates `A*eta = N / (t * I)` when a calibration and database match are
  available.
- Produces optional fit, ROI-expansion, chi-square, and systematic-error plots.
- Continues processing other ROIs when an individual fit fails, recording the
  failure in the output.

## Installation

Clone the repository and install the runtime dependencies:

```bash
git clone https://github.com/csopetap/MCA_data_analysis.git
cd MCA_data_analysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install numpy pandas scipy matplotlib
```

The package can be imported directly from the repository root. There is
currently no packaging metadata, so installation with `pip install .` is not
available.

## Input data

Place one or more `.mca` files in an input directory. The reader expects the
file to contain:

```text
LIVE_TIME = 0 123.45
<<ROI>>
100 140
650 675
<<DATA>>
...
<<END>>
```

The exact spacing and header text may vary, but the ROI section must contain
pairs of integer channel boundaries and the data section must contain one
integer count per line. Channel numbers are the indices of the count array.

The repository includes `mca_pipeline/Intensity_Database.csv`, a database of
decay-chain, isotope, energy, intensity, and intensity-error values. Its
columns are read by position, so the first five columns must be:

1. Decay chain
2. Isotope
3. Energy in keV
4. Intensity in percent
5. Intensity error in percent

## Basic usage

```python
from mca_pipeline import analyze_folder

results = analyze_folder(
    input_folder="data/mca",
    output_folder="results",
    calibration_parameters=(-2.15, 0.352),
    database_path="mca_pipeline/Intensity_Database.csv",
    decay_chains=["238U", "232Th"],
    interactive=False,
)

for filename, table in results.items():
    print(filename)
    print(table[["ROI_input", "Energy measured (keV)", "Net area", "fit_success"]])
```

`calibration_parameters` is a two-element tuple `(a, b)` for
`E = a + b * channel`. Use `(-1, -1)` or `None` to disable calibration and
database matching. In that mode, energy, intensity, and `A*eta` fields are
empty (`NaN`) and `line_matched` is `False`.

When multiple database lines are within the matching tolerance,
`interactive=True` prompts for a choice. Use `interactive=False` for scripts
and automated processing; the closest line is selected with a warning.

## Output

For each input file, the pipeline writes:

```text
results/
├── CSV_Outputs/
│   ├── Result_spectrum_1.csv
│   └── Result_spectrum_2.csv
└── Plot_Outputs/
    ├── Fit_plots/
    ├── ROI_plots/
    ├── Chi2_plots/
    └── Syst_err_plots/
```

Plot generation can be disabled or redirected with `plot_folder` and the
`plot_ROI_expand`, `plotFit`, `plotChi2`, and `plotSystError` arguments.

The standard CSV columns are:

| Column | Description |
| --- | --- |
| `LIVE_TIME` | Acquisition live time read from the `.mca` file |
| `ROI_input` | Original ROI, formatted as `[low, high]` |
| `Energy measured (keV)` | Calibrated fitted centroid |
| `Energy (keV)` | Matched database energy |
| `Intensity`, `Intensity error` | Matched database intensity values |
| `line_matched` | Whether a database line was assigned |
| `Net area`, `Net area error` | Fitted peak area and combined uncertainty |
| `FWHM (Ch)`, `dFWHM (Ch)` | FWHM and its uncertainty in channels |
| `A*eta`, `A*eta Error` | Activity-efficiency product and uncertainty |
| `fit_success` | Whether the ROI fit passed all checks |

Set `include_debug_columns=True` to also include the recommended ROI,
centroid, total counts, statistical/systematic area errors, matched isotope
and decay chain, and `fail_reason`.

## Main API

```python
analyze_folder(
    input_folder,
    output_folder,
    calibration_parameters=(-1, -1),
    decay_chains=None,
    database_path="Intensity_Database.csv",
    tolerance_keV=5.0,
    plot_folder=None,
    second_der_signif_toler=2,
    first_der_signif_toler=1.5,
    search_max_low=20,
    search_max_high=20,
    wROI_window=10,
    shift_ratio=0.3,
    plot_ROI_expand=True,
    plotFit=True,
    plotChi2=True,
    plotSystError=True,
    plotLang="eng",
    interactive=True,
    include_debug_columns=False,
    max_relative_error=1.0,
)
```

Important behavior:

- Input files are processed in sorted filename order.
- Results are written immediately after each file finishes.
- A missing `.mca` file raises `FileNotFoundError`.
- `plotLang` must be `"eng"` or `"hun"`.
- `max_relative_error` marks a fit unsuccessful when
  `Net area error / abs(Net area)` exceeds the limit. Set it to `None` to
  disable this check.
- `decay_chains` accepts one chain name, a list of names, or `None` for the
  full database. Unknown chain names raise `ValueError`.

## Programmatic results

The function returns a dictionary mapping each processed `.mca` filename to
the same pandas `DataFrame` written to disk:

```python
results = analyze_folder("data/mca", "results", calibration_parameters=None)
first_result = results["spectrum_1.mca"]

successful = first_result[first_result["fit_success"]]
successful.to_excel("successful_peaks.xlsx", index=False)
```

## License

This project is released under the [MIT License](LICENSE).

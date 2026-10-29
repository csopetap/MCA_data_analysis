"""ROI expansion, peak fitting and error estimation."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from . import plotting
from .models import (FWHM_FACTOR, get_gradient, get_peak_area, peak_plus_bkg_function,
                     second_der_signif_anal, bkg_slope, bkg_slope_err)


def fit_single_peak_single_ROI(chData, recROI):
    """Fits one peak in the given ROI. Returns popt, pcov. No plotting."""
    lo, hi = int(recROI[0]), int(recROI[1])

    # fit parameter bounds and some original parameter estimates
    fit_bounds = ([0, lo, 0.01, 0, 0.01, 0, -np.inf, 0],
                  [np.inf, hi, 50, 30, 100, 0.5, np.inf, np.inf])
    estimated_slope = (chData[hi] - chData[lo]) / (hi - lo)
    est_C = (chData[lo] + chData[hi]) / 2.

    window = chData[lo:hi + 1]
    # exp() can overflow for extreme trial parameters during the optimisation; that is harmless
    with np.errstate(over="ignore", invalid="ignore"):
        popt, pcov = curve_fit(
            peak_plus_bkg_function,
            np.arange(lo, hi + 1, dtype=float),
            window,
            sigma=np.max([np.sqrt(window), np.ones(hi + 1 - lo)], axis=0),
            absolute_sigma=True,
            p0=[
                int(np.max(window)),
                int(lo + np.where(window == np.max(window))[0][0]),
                3, 0.05, 1.5, 0.001, float(estimated_slope), float(np.max([est_C, 0]))
            ],
            bounds=fit_bounds, maxfev=1e4)
    return popt, pcov


def expand_ROI(chData, input_ROI, wROI_search_max_low=20, wROI_search_max_high=20, wROI_window = 10,
               second_derivative_signif_tolerance=2, first_derivative_signif_tolerance = 1.5):
    """
    Expands the input ROI as long as the measured second-derivative significance does not
    exceed the one predicted from the initial fit (plus a tolerance).

    Returns the recommended ROI [low, high] and a dict with the arrays needed for plotting.
    """
    wROI_window = 10  # How many channels are we looking at the same time to check the significance
    k = round(wROI_window/2.5)  # How many channels have to exceed the second derivative threshold to stop
    # Initial fit with the input ROI to get an estimate of the fit parameters
    init_fit_params, init_fit_cov = fit_single_peak_single_ROI(chData, input_ROI)

    x_arr = np.arange(len(chData))
    bkg_slope_arr = bkg_slope(x_arr, *init_fit_params)
    bkg_slope_err_arr = bkg_slope_err(x_arr, init_fit_params, init_fit_cov)
    # Analytical second derivative significance based on the initial parameters
    S_list = second_der_signif_anal(x_arr, *init_fit_params)

    second_derivative = chData[2:] - 2*chData[1:-1] + chData[:-2]
    second_derivative_error = np.max([np.sqrt(chData[:-2] + 4*chData[1:-1] + chData[2:]),
                                      np.ones(len(second_derivative))], axis=0)
    second_derivative_significance = np.abs(second_derivative) / second_derivative_error
    # np.inf on both ends so indexing is not shifted
    second_derivative_significance = np.concatenate(([np.inf], second_derivative_significance, [np.inf]))

    # Expanding ROI on the left
    ROI_low_curr = input_ROI[0]
    while True:
        cand = ROI_low_curr - 1  # Current candidate

        if cand < 1 or ROI_low_curr <= input_ROI[0] - wROI_search_max_low:
            print(f'For {input_ROI} => Stopping on the left: Window reached')
            break

        # Floored at 0 so we don't run into the low bound of the spectrum
        window_low = max(cand - wROI_window + 1, 0)
        exceed_second_deriv = (second_derivative_significance[window_low:cand + 1]
                  > S_list[window_low:cand + 1] + second_derivative_signif_tolerance)
        # If there are more than k channels exceeding the threshold
        if np.sum(exceed_second_deriv) >= k:
            print(f'For {input_ROI} => Stopping on the left: Second derivative too high')
            break
        first_deriv_arr = []
        for ch in range(1, wROI_window+1):
            first_deriv_arr.append((chData[cand - ch] - chData[cand])/ch)
        first_deriv_avg = np.mean(first_deriv_arr)
        first_deriv_std = np.std(first_deriv_arr, ddof = 1)

        B = bkg_slope_arr[cand]
        dB = bkg_slope_err_arr[cand]

        first_deriv_signif = np.abs(B - first_deriv_avg)/(np.sqrt(dB**2 + first_deriv_std**2))
        if first_deriv_signif > first_derivative_signif_tolerance:
            print(f'For {input_ROI} => Stopping on the left: First derivative too high')
            break
        ROI_low_curr = cand

    # Expanding ROI on the right
    ROI_high_curr = input_ROI[1]
    while True:
        cand = ROI_high_curr + 1

        if cand > len(chData) - 2 or ROI_high_curr >= input_ROI[1] + wROI_search_max_high:
            print(f'For {input_ROI} => Stopping on the right: Window reached')
            break

        exceed = (second_derivative_significance[cand:cand + wROI_window]
                  > S_list[cand:cand + wROI_window] + second_derivative_signif_tolerance)
        if np.sum(exceed) >= k:
            print(f'For {input_ROI} => Stopping on the right: Second derivative too high')
            break
        first_deriv_arr = []
        for ch in range(1, wROI_window+1):
            first_deriv_arr.append((chData[cand + ch] - chData[cand])/ch)
        first_deriv_avg = np.mean(first_deriv_arr)
        first_deriv_std = np.std(first_deriv_arr, ddof = 1)

        B = bkg_slope_arr[cand]
        dB = bkg_slope_err_arr[cand]

        first_deriv_signif = np.abs(B - first_deriv_avg)/(np.sqrt(dB**2 + first_deriv_std**2))
        if first_deriv_signif > first_derivative_signif_tolerance:
            print(f'For {input_ROI} => Stopping on the right: First derivative too high')
            break
        ROI_high_curr = cand

    diag = {"S_list": S_list, "significance": second_derivative_significance}
    return [int(ROI_low_curr), int(ROI_high_curr)], diag


def syst_error_with_variation(chData, ROI_in, ROI_rec, shift_ratio=0.3, nominal_area=None):
    """
    Systematic error of the peak area from shrinking the recommended ROI towards the input ROI.
    Returns sigma_syst, the list of areas (areas[0] is the nominal fit), n_low, n_high.
    """
    # how far the recommended ROI extends beyond the input ROI on each side
    d_low = ROI_in[0] - ROI_rec[0]
    d_high = ROI_rec[1] - ROI_in[1]

    def n_steps(d):
        if d <= 1:
            return 0  # no room to shrink without touching the input ROI
        return min(max(1, round(shift_ratio * d)), d - 1)

    n_low, n_high = n_steps(d_low), n_steps(d_high)
    step_low = max(1, n_low // 10)
    step_high = max(1, n_high // 10)

    areas = []
    for i in range(0, n_low + 1, step_low):
        for j in range(0, n_high + 1, step_high):
            if i == 0 and j == 0 and nominal_area is not None:
                areas.append(nominal_area)
                continue
            roi = [ROI_rec[0] + i, ROI_rec[1] - j]
            try:
                popt, _ = fit_single_peak_single_ROI(chData, roi)
                areas.append(get_peak_area(*popt))
            except Exception:
                if i == 0 and j == 0:
                    raise  # nominal fit must work
                continue  # skip failed variations

    areas = np.array(areas)
    if len(areas) < 2:
        return 0.0, areas, n_low, n_high, step_low, step_high
    A0 = areas[0]
    sigma_syst = np.sqrt(np.sum((areas[1:] - A0) ** 2) / (len(areas) - 1))
    return sigma_syst, areas, n_low, n_high, step_low, step_high


def _analyse_peak(chData, ROI, file_stem, second_der_signif_toler, first_der_signif_toler, search_max_low, search_max_high, wROI_window,
                  shift_ratio, plot_cfg, max_relative_error):
    rec_ROI, diag = expand_ROI(chData, ROI, search_max_low, search_max_high, wROI_window, second_der_signif_toler, first_der_signif_toler)
    if plot_cfg is not None and plot_cfg.roi_expand:
        plotting.plot_roi_expansion(
            chData, ROI, rec_ROI, diag,
            plot_cfg.roi_dir / f"{file_stem}_ROI_{ROI[0]}_{ROI[1]}_expand_ROI.pdf")

    popt, pcov = fit_single_peak_single_ROI(chData, rec_ROI)
    if not np.all(np.isfinite(pcov)):
        raise RuntimeError("covariance of the fit could not be estimated")
    perr = np.sqrt(np.diag(pcov))

    net_area = get_peak_area(*popt)
    grad = get_gradient(*popt)
    stat_error = np.sqrt(np.dot(grad, np.matmul(pcov, grad)))
    if not (np.isfinite(net_area) and np.isfinite(stat_error)):
        raise RuntimeError("net area or its statistical error is not finite")

    if plot_cfg is not None and plot_cfg.fit:
        plotting.plot_fit(chData, rec_ROI, popt,
                          plot_cfg.fit_dir / f"{file_stem}_ROI_{ROI[0]}_{ROI[1]}_fit.pdf",
                          plot_cfg.lang)
    if plot_cfg is not None and plot_cfg.chi2:
        plotting.plot_chi2(chData, rec_ROI, popt,
                           plot_cfg.chi2_dir / f"{file_stem}_ROI_{ROI[0]}_{ROI[1]}_chi2.pdf")

    syst_error, areas, n_low, n_high, step_low, step_high = syst_error_with_variation(
        chData, ROI, rec_ROI, shift_ratio, nominal_area=net_area)
    if plot_cfg is not None and plot_cfg.syst_err:
        plotting.plot_syst_error(
            areas, syst_error, n_low, n_high, step_low, step_high,
            plot_cfg.syst_dir / f"{file_stem}_ROI_{ROI[0]}_{ROI[1]}_syst_err.pdf")

    if syst_error > stat_error * 3:
        print(f"Warning! Big systematical error on peak {ROI}")

    full_error = np.sqrt(syst_error**2 + stat_error**2)
    # A fit that converged but has no meaningful area is also a failed peak
    if max_relative_error is not None:
        rel_error = full_error / abs(net_area) if net_area != 0 else np.inf
        if not rel_error <= max_relative_error:
            raise RuntimeError(f"relative net area error {rel_error:.3g} exceeds {max_relative_error}")

    return {
        "ROI_input": f"[{ROI[0]}, {ROI[1]}]",
        "ROI_recommended": f"[{rec_ROI[0]}, {rec_ROI[1]}]",
        "Centroid": popt[1],
        "Total Area": float(np.sum(chData[rec_ROI[0]:rec_ROI[1] + 1])),  # kept for debugging
        "Net area": net_area,
        "Net area error": full_error,
        "Net area stat error": stat_error,
        "Net area syst error": syst_error,
        "FWHM (Ch)": FWHM_FACTOR * popt[2],
        "dFWHM (Ch)": FWHM_FACTOR * perr[2],
        "fit_success": True,
        "fail_reason": "",
    }


def expand_and_fit_all_peaks(chData, input_ROI_list, file_stem="", second_der_signif_toler=2, 
                            first_der_signif_toler = 1.5,
                            search_max_low=20, search_max_high=20, wROI_window = 10, shift_ratio=0.3, plot_cfg=None,
                            max_relative_error=1.0):
    """
    Expands and fits every ROI. A peak that fails (exception, non-finite result, or a relative
    net area error above max_relative_error; None disables that last check) is flagged
    (fit_success=False, NaN values) and the rest continues.
    Returns a DataFrame with one row per input ROI.
    """
    # TODO: Implement unique border for each ROI so they don't expand into each other
    rows = []
    for ROI in input_ROI_list:
        ROI = [int(ROI[0]), int(ROI[1])]
        try:
            rows.append(_analyse_peak(chData, ROI, file_stem, second_der_signif_toler,
                                      first_der_signif_toler,
                                      search_max_low, search_max_high, wROI_window, shift_ratio, plot_cfg,
                                      max_relative_error))
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"Warning: fit failed for ROI {ROI} ({file_stem}): {reason}")
            rows.append({
                "ROI_input": f"[{ROI[0]}, {ROI[1]}]", "ROI_recommended": "",
                "Centroid": np.nan, "Total Area": np.nan, "Net area": np.nan,
                "Net area error": np.nan, "Net area stat error": np.nan,
                "Net area syst error": np.nan, "FWHM (Ch)": np.nan, "dFWHM (Ch)": np.nan,
                "fit_success": False, "fail_reason": reason,
            })
    return pd.DataFrame(rows)

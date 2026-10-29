"""Matching the measured peak energies to the database lines."""
import numpy as np
import pandas as pd

# The database columns are read by position, so the header language doesn't matter:
# decay chain, isotope, energy (keV), intensity (%), intensity error
DB_COLUMNS = ["Decay chain", "Isotope", "Energy", "Intensity", "Intensity error"]


class MatchingAborted(RuntimeError):
    """Raised when the user chooses to stop at the interactive question."""


def load_database(path, decay_chains=None):
    """
    Reads the database and keeps only the lines of the given decay chain(s)
    (a string or a list of strings, e.g. ["238U", "232Th"]). None keeps everything.
    """
    db = pd.read_csv(path, skipinitialspace=True)
    if db.shape[1] < 5:
        raise ValueError(f"{path}: expected at least 5 columns, found {db.shape[1]}")
    db = db.iloc[:, :5].copy()
    db.columns = DB_COLUMNS
    for col in ("Energy", "Intensity", "Intensity error"):
        db[col] = pd.to_numeric(db[col], errors="coerce")
    db = db.dropna(subset=["Energy"])
    for col in ("Decay chain", "Isotope"):
        db[col] = db[col].astype(str).str.strip()

    if decay_chains is not None:
        if isinstance(decay_chains, str):
            decay_chains = [decay_chains]
        wanted = [str(c).strip() for c in decay_chains]
        available = sorted(db["Decay chain"].unique())
        unknown = [c for c in wanted if c not in available]
        if unknown:
            raise ValueError(f"Unknown decay chain(s) {unknown}. Available: {available}")
        db = db[db["Decay chain"].isin(wanted)]
    return db.reset_index(drop=True)


def _resolve_conflict(energy, label, candidates, tolerance, interactive):
    """Several lines are in tolerance: ask which one to use. Returns a row, or None to leave it empty."""
    shown = candidates[["Decay chain", "Isotope", "Energy", "Intensity", "Diff (keV)"]]
    print(f".\nWarning: CLOSE PEAKS - {energy:.3f} keV ({label}) can be assigned to "
          f"{len(candidates)} database lines within +/-{tolerance} keV:")
    print(shown.to_string(), "\n", sep="")

    if not interactive:
        print("Non-interactive mode: using the closest line.")
        return candidates.iloc[0]

    prompt = ("a) Choose the closest anyway (Enter)\n"
              f"b) Choose a different one - write the index (0-{len(candidates) - 1})\n"
              "c) Continue leaving the row empty (c)\n"
              "d) STOP the script (s)\n"
              "Please write your answer and hit Enter: ")
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("", "a", "a)"):
            return candidates.iloc[0]
        if answer.isdigit() and int(answer) < len(candidates):
            return candidates.iloc[int(answer)]
        if answer in ("c", "c)"):
            return None
        if answer in ("d", "d)", "s", "stop"):
            raise MatchingAborted("Stopped by the user at the database matching question.")
        print("Incorrect answer. Try again.")


def match_peaks(energies, labels, db, tolerance=4.0, interactive=True):
    """
    Matches every measured energy to the database (already filtered to the wanted decay chains).

    - exactly one line within the tolerance: it is used
    - several lines within the tolerance: the user is asked (or the closest is used if not interactive)
    - none within the tolerance: a warning with the closest line is printed and the row stays NaN

    NaN energies (failed fits) are skipped. Returns a DataFrame aligned with `energies`;
    its `line_matched` column is True where a database line was assigned.
    """
    n = len(energies)
    result = pd.DataFrame({
        "Energy (keV)": np.full(n, np.nan),
        "Intensity": np.full(n, np.nan),
        "Intensity error": np.full(n, np.nan),
        "Isotope": [None] * n,
        "Decay chain": [None] * n,
        "line_matched": np.zeros(n, dtype=bool),
    })
    chains = ", ".join(sorted(db["Decay chain"].unique()))

    for k, (energy, label) in enumerate(zip(energies, labels)):
        if not np.isfinite(energy):
            continue

        ranked = (db.assign(**{"Diff (keV)": (db["Energy"] - energy).abs()})
                    .sort_values("Diff (keV)", kind="stable").reset_index(drop=True))
        candidates = ranked[ranked["Diff (keV)"] <= tolerance]

        if candidates.empty:
            closest = ranked.iloc[0]
            print(f"Warning: NOT FOUND - no database line within +/-{tolerance} keV of "
                  f"{energy:.3f} keV ({label}; chains: {chains}). Closest line: "
                  f"{closest['Isotope']} ({closest['Decay chain']}) {closest['Energy']:.3f} keV, "
                  f"difference {closest['Diff (keV)']:.3f} keV.")
            continue

        if len(candidates) == 1:
            chosen = candidates.iloc[0]
        else:
            chosen = _resolve_conflict(energy, label, candidates, tolerance, interactive)
            if chosen is None:
                continue

        result.loc[k, "Energy (keV)"] = chosen["Energy"]
        result.loc[k, "Intensity"] = chosen["Intensity"]
        result.loc[k, "Intensity error"] = chosen["Intensity error"]
        result.loc[k, "Isotope"] = chosen["Isotope"]
        result.loc[k, "Decay chain"] = chosen["Decay chain"]
        result.loc[k, "line_matched"] = True

    return result

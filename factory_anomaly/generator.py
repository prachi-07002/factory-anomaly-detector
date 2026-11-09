"""Synthetic sensor data for a CNC / robot manufacturing cell.

Each sample is one short monitoring window summarised into 6 features:

    0  vib_rms       - vibration RMS (g)
    1  vib_kurtosis  - vibration kurtosis (impulsiveness)
    2  temperature   - spindle / motor temperature (deg C)
    3  torque_mean   - spindle torque mean (Nm)
    4  torque_std    - spindle torque variability (Nm)
    5  vib_peak      - peak vibration in the window (g)

Normal operation is a correlated multivariate Gaussian. Three fault
modes shift the feature distribution in physically plausible ways:

    bearing_wear   - vibration RMS and kurtosis creep up, temperature
                     rises a little (progressive degradation).
    overheating    - temperature jumps while torque *drops* (the motor
                     derates as it overheats), breaking the normal
                     temperature/torque correlation.
    tool_breakage  - sudden torque spike and a sharp vibration peak
                     (an abrupt event, not a slow drift).

Only NumPy is used. Everything is seeded for reproducibility.
"""

import numpy as np

FEATURE_NAMES = [
    "vib_rms",
    "vib_kurtosis",
    "temperature",
    "torque_mean",
    "torque_std",
    "vib_peak",
]

FAULT_TYPES = ["bearing_wear", "overheating", "tool_breakage"]

# Normal operating point (means) and per-feature standard deviations.
_NORMAL_MEAN = np.array([0.45, 3.0, 62.0, 18.0, 1.2, 1.10])
_NORMAL_STD = np.array([0.08, 0.35, 2.5, 1.5, 0.25, 0.22])

# Correlation between features in normal operation (vibration features
# move together, temperature correlates weakly with torque).
_CORR = np.array([
    [1.00, 0.55, 0.20, 0.30, 0.45, 0.70],
    [0.55, 1.00, 0.10, 0.15, 0.35, 0.60],
    [0.20, 0.10, 1.00, 0.45, 0.10, 0.15],
    [0.30, 0.15, 0.45, 1.00, 0.50, 0.25],
    [0.45, 0.35, 0.10, 0.50, 1.00, 0.40],
    [0.70, 0.60, 0.15, 0.25, 0.40, 1.00],
])

# Mean shift (in units of _NORMAL_STD) applied by each fault.
# Note overheating also *lowers* torque: an overheating motor derates
# (loses efficiency), so hot + weak torque breaks the normal
# temperature/torque correlation - exactly what makes it detectable.
_FAULT_SHIFTS = {
    "bearing_wear":  np.array([2.8, 3.2, 1.2, 0.4, 1.2, 2.5]),
    "overheating":   np.array([0.5, 0.3, 3.8, -0.8, 0.5, 0.6]),
    "tool_breakage": np.array([3.0, 2.0, 0.5, 3.5, 3.0, 4.0]),
}
# Faults are also noisier than normal operation (extra std multiplier).
_FAULT_NOISE = {"bearing_wear": 1.3, "overheating": 1.2, "tool_breakage": 1.6}


def _normal_cov():
    d = _NORMAL_STD
    return _CORR * np.outer(d, d)


def _sample_normal(rng, n):
    return rng.multivariate_normal(_NORMAL_MEAN, _normal_cov(), size=n)


def _sample_fault(rng, n, fault):
    mean = _NORMAL_MEAN + _FAULT_SHIFTS[fault] * _NORMAL_STD
    cov = _normal_cov() * (_FAULT_NOISE[fault] ** 2)
    return rng.multivariate_normal(mean, cov, size=n)


def generate_dataset(n_normal=1000, n_anomaly=200, seed=42,
                     fault_mix=None):
    """Generate a labelled dataset.

    Returns (X, y, fault_labels) where X is (n, 6), y is 0 = normal /
    1 = anomaly, and fault_labels is "" for normal rows and the fault
    name for anomalous rows. Faults are split evenly by default.
    """
    rng = np.random.default_rng(seed)
    if fault_mix is None:
        fault_mix = {f: 1.0 / len(FAULT_TYPES) for f in FAULT_TYPES}

    X_norm = _sample_normal(rng, n_normal)
    y_norm = np.zeros(n_normal, dtype=int)
    f_norm = np.array([""] * n_normal)

    xs, ys, fs = [X_norm], [y_norm], [f_norm]
    for fault, frac in fault_mix.items():
        n_f = int(round(n_anomaly * frac))
        if n_f <= 0:
            continue
        xs.append(_sample_fault(rng, n_f, fault))
        ys.append(np.ones(n_f, dtype=int))
        fs.append(np.array([fault] * n_f))

    X = np.vstack(xs)
    y = np.concatenate(ys)
    faults = np.concatenate(fs)

    # Shuffle so normal / fault rows are interleaved.
    perm = rng.permutation(len(X))
    return X[perm], y[perm], faults[perm]


def generate_stream(n_windows=500, seed=7, fault_at=300,
                    fault="bearing_wear", severity=1.0):
    """Generate a time-ordered stream: normal, then a fault onset.

    Useful for the demo: shows a detector flagging the moment a fault
    begins. Returns (X, y, fault_labels) in time order (not shuffled).
    severity scales the fault mean shift (0.5 = early / subtle fault).
    """
    rng = np.random.default_rng(seed)
    X_norm = _sample_normal(rng, fault_at)
    mean = _NORMAL_MEAN + _FAULT_SHIFTS[fault] * _NORMAL_STD * severity
    cov = _normal_cov() * (_FAULT_NOISE[fault] ** 2)
    X_fault = rng.multivariate_normal(mean, cov, size=n_windows - fault_at)
    X = np.vstack([X_norm, X_fault])
    y = np.array([0] * fault_at + [1] * (n_windows - fault_at))
    faults = np.array([""] * fault_at + [fault] * (n_windows - fault_at))
    return X, y, faults


def standardize(X_train, X_test):
    """Z-score using train mean/std. Returns (Xtr, Xte, mean, std)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std

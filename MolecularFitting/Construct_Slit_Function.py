import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CALIBRATION_FILENAME = "09_04_2026.spa"
CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "Slit_Functions", "SPA", CALIBRATION_FILENAME)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Slit_Functions")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "Slit_Functions", "Figures")

# Spectral extraction region
CENTRAL_WAVELENGTH = 435.833  # Target emission line (nm)
DELTA_WAVELENGTH = 1.0        # Range (+/- nm)
# =========================================================


def extract_central_line(file_path, output_dir, figures_dir, center_wl, delta_wl):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load 2-column ASCII data, skipping header row
    data = np.loadtxt(file_path, skiprows=1)
    raw_wavelength = data[:, 0]
    raw_intensity = data[:, 1]

    # Crop to range [center_wl - delta_wl, center_wl + delta_wl]
    min_wl = center_wl - delta_wl
    max_wl = center_wl + delta_wl
    mask = (raw_wavelength >= min_wl) & (raw_wavelength <= max_wl)

    wavelength = raw_wavelength[mask]
    intensity = raw_intensity[mask]

    if len(wavelength) == 0:
        raise ValueError(f"No spectral data found within range {min_wl:.3f} nm to {max_wl:.3f} nm.")

    # Locate peak intensity within cropped range
    peak_idx = np.argmax(intensity)
    peak_wl = wavelength[peak_idx]

    # Center wavelength scale around the peak (peak = 0.0 nm)
    relative_wavelength = wavelength - peak_wl

    # Normalize peak intensity to 1.0
    norm_intensity = intensity / intensity[peak_idx] if intensity[peak_idx] != 0 else intensity

    # Base filename without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # ---------------------------------------------------------
    # 1. Save TXT Data
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}.txt")

    output_data = np.column_stack((relative_wavelength, norm_intensity))
    np.savetxt(
        output_path,
        output_data,
        fmt="%.6f",
        delimiter="\t",
        header=(
            f"Target_Line: {center_wl} nm (+/- {delta_wl} nm)\n"
            f"Peak_Wavelength: {peak_wl:.4f} nm\n"
            f"Relative_Wavelength\tNormalized_Intensity"
        ),
        comments="#"
    )

    # ---------------------------------------------------------
    # 2. Plot & Save Figure
    # ---------------------------------------------------------
    os.makedirs(figures_dir, exist_ok=True)
    figure_path = os.path.join(figures_dir, f"{base_name}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(relative_wavelength, norm_intensity, color="darkblue", linewidth=1.5, label="Slit Function")
    plt.axvline(0, color="red", linestyle="--", alpha=0.7, label=f"Peak ({peak_wl:.3f} nm)")
    
    plt.title(f"Slit Function Profile: {base_name}\nTarget: {center_wl} nm (±{delta_wl} nm)")
    plt.xlabel("Relative Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(figure_path, dpi=300)
    print(f"Extracted range: {min_wl:.3f} nm to {max_wl:.3f} nm")
    print(f"Peak line located at: {peak_wl:.4f} nm")
    print(f"Saved text data: {output_path}")
    print(f"Saved figure: {figure_path}")
    
    plt.show()


if __name__ == "__main__":
    extract_central_line(
        CALIBRATION_FILE, 
        OUTPUT_DIR, 
        FIGURES_DIR,
        center_wl=CENTRAL_WAVELENGTH, 
        delta_wl=DELTA_WAVELENGTH
    )
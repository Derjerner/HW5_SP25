# region imports
import hw5a as pta  # Import Moody Diagram and friction factor functions
import random as rnd
import numpy as np
from matplotlib import pyplot as plt


# endregion

# region functions

def interpolate_transition_f(Re, rr):
    """
    Interpolate friction factor in transition region (2000 < Re < 4000) with randomness.
    :param Re: Reynolds number
    :param rr: relative roughness
    :return: interpolated friction factor
    """
    f_lam = pta.ff(2000, rr, False)
    f_turb = pta.ff(4000, rr, True)
    t = (Re - 2000) / (4000 - 2000)  # Linear interpolation factor

    mean_f = (1 - t) * f_lam + t * f_turb  # Weighted mean
    std_f = 0.2 * mean_f  # Standard deviation = 20% of mean

    return max(0.008, rnd.gauss(mean_f, std_f))  # Ensure f is always > 0.008


def calculate_head_loss(D_in, eps_micro, Q_gpm):
    """
    Calculate head loss per foot based on user inputs.
    :param D_in: pipe diameter in inches
    :param eps_micro: roughness in micro-inches
    :param Q_gpm: flow rate in gallons/min
    :return: tuple of (hf/L, Re, f, rr)
    """
    # Constants
    g = 32.174  # ft/s^2 (gravity)
    nu = 1.41e-5  # kinematic viscosity of water at 60°F in ft²/s

    # Ensure valid input
    if D_in <= 0 or Q_gpm <= 0:
        raise ValueError("Pipe diameter and flow rate must be positive.")

    # Conversions
    D_ft = D_in / 12  # Convert inches to feet
    eps_ft = eps_micro * 1e-6 / 12  # Convert micro-inches to feet
    Q_cfs = Q_gpm / 448.831  # Convert GPM to ft³/s

    # Velocity calculation
    A = np.pi * (D_ft / 2) ** 2  # Pipe cross-sectional area (ft²)
    V = Q_cfs / A  # Flow velocity (ft/s)

    # Compute Reynolds number
    Re = V * D_ft / nu
    rr = eps_ft / D_ft  # Relative roughness

    # Determine friction factor
    if Re <= 2000:
        f = pta.ff(Re, rr, False)  # Laminar
    elif Re >= 4000:
        f = pta.ff(Re, rr, True)  # Turbulent (Colebrook)
    else:
        f = interpolate_transition_f(Re, rr)  # Transition

    # Compute head loss per foot
    hf_per_L = f * (V ** 2) / (2 * g * D_ft)

    return hf_per_L, Re, f, rr


def plotMoody(points=[]):
    """
    Produce Moody diagram with optional points from calculations.
    :param points: list of tuples (Re, f, is_transition) for plotting
    """
    # Reynolds number ranges
    ReValsCB = np.logspace(np.log10(4000), 8, 100)
    ReValsL = np.logspace(np.log10(600.0), np.log10(2000.0), 20)
    ReValsTrans = np.logspace(np.log10(2000.0), np.log10(4000.0), 20)

    rrVals = np.array([0, 1E-6, 5E-6, 1E-5, 5E-5, 1E-4, 2E-4, 4E-4, 6E-4, 8E-4,
                       1E-3, 2E-3, 4E-3, 6E-3, 8E-8, 1.5E-2, 2E-2, 3E-2, 4E-2, 5E-2])

    # Calculate friction factors
    ffLam = np.array([pta.ff(Re, 0) for Re in ReValsL])
    ffTrans = np.array([pta.ff(Re, 0) for Re in ReValsTrans])
    ffCB = np.array([[pta.ff(Re, rr, True) for Re in ReValsCB] for rr in rrVals])

    # Plot Moody diagram
    plt.figure(figsize=(10, 6))
    plt.loglog(ReValsL, ffLam, 'b-', linewidth=2, label="Laminar")
    plt.loglog(ReValsTrans, ffTrans, 'b--', linewidth=2, label="Transition")

    for nRelR in range(len(ffCB)):
        plt.loglog(ReValsCB, ffCB[nRelR], 'k')
        plt.annotate(xy=(ReValsCB[-1], ffCB[nRelR][-1]), text=f"{rrVals[nRelR]:.1e}")

    # Plot calculated points
    for Re, f, is_trans in points:
        marker = '^' if is_trans else 'o'  # Triangle for transition, circle otherwise
        plt.plot(Re, f, marker, markersize=10, markeredgecolor='red', markerfacecolor='none')

    # Formatting
    plt.xlim(600, 1E8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number $Re$", fontsize=16)
    plt.ylabel(r"Friction factor $f$", fontsize=16)
    plt.grid(which='both', linestyle='-', alpha=0.5)
    plt.legend()
    plt.show()


def main():
    """
    Main function to handle user input and program flow.
    """
    points = []

    while True:
        try:
            D_in = float(input("Enter pipe diameter (inches): "))
            eps_micro = float(input("Enter pipe roughness (micro-inches): "))
            Q_gpm = float(input("Enter flow rate (gallons/min): "))

            hf_per_L, Re, f, rr = calculate_head_loss(D_in, eps_micro, Q_gpm)
            is_transition = 2000 < Re < 4000

            print(f"\nHead loss per foot: {hf_per_L:.6f} ft/ft")
            print(f"Reynolds number: {Re:.2f}")
            print(f"Friction factor: {f:.4f}")
            print(f"Flow regime: {'Transition' if is_transition else 'Laminar' if Re <= 2000 else 'Turbulent'}")

            points.append((Re, f, is_transition))
            plotMoody(points)

            if input("\nCalculate another case? (y/n): ").lower() != 'y':
                break

        except ValueError:
            print("⚠ Please enter valid numerical values.")
        except Exception as e:
            print(f"⚠ Error: {e}")


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion

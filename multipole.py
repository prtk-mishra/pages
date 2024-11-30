import numpy as np
import h5py

# Load data from the .mat file (These packages are needed)

matfile = 'E:/Multipole/ENxyzf.mat' #Give full path where your exported data is
with h5py.File(matfile, 'r') as f:
    Ex = np.array(f['Ex'])  # Electric field component Ex
    Ey = np.array(f['Ey'])  # Electric field component Ey
    Ez = np.array(f['Ez'])  # Electric field component Ez
    n_x = np.array(f['n_x'])  # Refractive index component along x
    n_y = np.array(f['n_y'])  # Refractive index component along y
    n_z = np.array(f['n_z'])  # Refractive index component along z
    x = np.array(f['x'])      # Spatial grid in x-direction
    y = np.array(f['y'])      # Spatial grid in y-direction
    z = np.array(f['z'])      # Spatial grid in z-direction
    f_vals = np.array(f['f'])  # Frequency values (Hz)

# Verify shapes of loaded arrays (This just checking the shapes...not needed in multipole calculations)
print(f"Shapes: Ex={Ex.shape}, Ey={Ey.shape}, Ez={Ez.shape}, "
      f"n_x={n_x.shape}, n_y={n_y.shape}, n_z={n_z.shape}, "
      f"x={x.shape}, y={y.shape}, z={z.shape}, f={f_vals.shape}")

# Convert electric fields to complex numbers
Ex_complex = Ex['real'] + 1j * Ex['imag']
Ey_complex = Ey['real'] + 1j * Ey['imag']
Ez_complex = Ez['real'] + 1j * Ez['imag']

# Extract refractive indices
n_x = n_x['real']
n_y = n_y['real']
n_z = n_z['real']

# Angular frequency array (rad/s)
omega = 2 * np.pi * f_vals  # Shape: (31,)

# Constants
epsilon_0 = 8.854e-12  # Permittivity of free space (F/m)

# Define function to compute current densities
def compute_current_density(Ex, Ey, Ez, n_x, n_y, n_z, omega):
    """
    Compute current densities Jx, Jy, Jz from electric fields and refractive indices.

    Parameters:
        Ex, Ey, Ez: Complex electric field components (4D arrays)
        n_x, n_y, n_z: Refractive indices (4D arrays)
        omega: Angular frequency (1D array)

    Returns:
        Jx, Jy, Jz: Current density components (4D arrays)
    """
    # Broadcast omega to match field shapes
    
    omega1 = omega[0, :]  # Flatten omega to (31,) from shape (1, 31)

    # Ex, Ey, Ez, n_x, n_y, n_z have shape (31, 22, 51, 51)
    # Broadcasting omega to match the shape of Ex, Ey, Ez, n_x, n_y, n_z
    omega_expanded = omega1[:, None, None, None]  # Shape will be (31, 1, 1, 1)
    omega_expanded = np.broadcast_to(omega_expanded, Ex.shape)  # Broadcast to (31, 22, 51, 51)

    # Compute current densities (J from E)
    Jx = -1j * omega_expanded * epsilon_0 * (n_x**2 - 1) * Ex
    Jy = -1j * omega_expanded * epsilon_0 * (n_y**2 - 1) * Ey
    Jz = -1j * omega_expanded * epsilon_0 * (n_z**2 - 1) * Ez

    return Jx, Jy, Jz

# Calculate current densities (call above funtion)
Jx, Jy, Jz = compute_current_density(Ex_complex, Ey_complex, Ez_complex, n_x, n_y, n_z, omega)

# Verify shapes of current densities (Just to veryfy the data shape not needed in multipole calc)
print(f"Shapes: Jx={Jx.shape}, Jy={Jy.shape}, Jz={Jz.shape}")

# Helper function to integrate 4D data to 1D (INTEGRATION Funtion)
def trapz4Dto1D(F, x, y, z):
    """
    Integrates a 4D array F over the x, y, and z dimensions using the trapezoidal rule.
    x, y, z are 1D arrays specifying the spatial grid.
    """
    # Ensure x, y, z are 1D arrays
    x, y, z = x.flatten(), y.flatten(), z.flatten()

    # Integrate over z (axis=1)
    Fz = np.trapz(F, z, axis=1)  # Reduces axis 1

    # Integrate over y (axis=2 of Fz after z integration)
    Fy = np.trapz(Fz, y, axis=1)  # Reduces axis 2

    # Integrate over x (axis=2 of Fy after y integration, originally axis 3 of F)
    Fx = np.trapz(Fy, x, axis=1)  # Reduces axis 3

    return Fx


# Constants for multipole moment calculations
c = 3e8  # Speed of light (m/s)
k = omega / c  # Wave number (rad/m)

# Function to calculate multipole moments
def calculate_moments(Jx, Jy, Jz, x, y, z, omega, k):
    """
    Calculates Cp, Cm, Cqe, Cqm given Jx, Jy, Jz (current densities),
    spatial grids x, y, z, and constants omega and k.
    """
    # Ensure x, y, z, and omega are 1D arrays
    x, y, z, omega = x.flatten(), y.flatten(), z.flatten(), omega.flatten()

    # Create 4D grids for spatial coordinates
    Z, Y, X, F = np.meshgrid(z, y, x, omega, indexing='ij')

    # Rearrange the grid arrays to match (frequency, z, x, y)
    X4d = np.transpose(X, (3, 0, 2, 1))
    Y4d = np.transpose(Y, (3, 0, 2, 1))
    Z4d = np.transpose(Z, (3, 0, 2, 1))

    # Compute dot product r · J
    rJ = X4d * Jx + Y4d * Jy + Z4d * Jz

    # Compute r · r
    rr = X4d**2 + Y4d**2 + Z4d**2

    # Cross product r x J
    rxJx = Y4d * Jz - Z4d * Jy
    rxJy = Z4d * Jx - X4d * Jz
    rxJz = X4d * Jy - Y4d * Jx

    # Constant for scattering cross-section
    const = k**4 / (6 * np.pi * epsilon_0**2)

    ### Electric dipole moment
    dpx = rJ * X4d - 2 * rr * Jx
    dpy = rJ * Y4d - 2 * rr * Jy
    dpz = rJ * Z4d - 2 * rr * Jz

    px = -1 / (1j * omega) * (trapz4Dto1D(Jx, x, y, z) + k**2 / 10 * trapz4Dto1D(dpx, x, y, z))
    py = -1 / (1j * omega) * (trapz4Dto1D(Jy, x, y, z) + k**2 / 10 * trapz4Dto1D(dpy, x, y, z))
    pz = -1 / (1j * omega) * (trapz4Dto1D(Jz, x, y, z) + k**2 / 10 * trapz4Dto1D(dpz, x, y, z))

    norm2_p = px * np.conj(px) + py * np.conj(py) + pz * np.conj(pz)
    Cp = const * norm2_p

    ### Magnetic dipole moment
    mx = 1 / 2 * trapz4Dto1D(rxJx, x, y, z)
    my = 1 / 2 * trapz4Dto1D(rxJy, x, y, z)
    mz = 1 / 2 * trapz4Dto1D(rxJz, x, y, z)

    norm2_m = mx * np.conj(mx) + my * np.conj(my) + mz * np.conj(mz)
    Cm = const * norm2_m / c**2

    # Calculate toroidal dipole T
    dTx = rJ * X4d - 2 * rr * Jx
    dTy = rJ * Y4d - 2 * rr * Jy
    dTz = rJ * Z4d - 2 * rr * Jz

    Tx = 1 / (10 * c) * trapz4Dto1D(dTx, x, y, z)
    Ty = 1 / (10 * c) * trapz4Dto1D(dTy, x, y, z)
    Tz = 1 / (10 * c) * trapz4Dto1D(dTz, x, y, z)

    norm2_T = Tx * np.conj(Tx) + Ty * np.conj(Ty) + Tz * np.conj(Tz)
    CT = const * abs((1j * k)**2 * norm2_T)

    # Calculate net electric dipole moment (p + ikT), pT
    pTx = px + 1j * k * Tx
    pTy = py + 1j * k * Ty
    pTz = pz + 1j * k * Tz

    norm2_pT = pTx * np.conj(pTx) + pTy * np.conj(pTy) + pTz * np.conj(pTz)
    CpT = const * norm2_pT


    ### Electric quadrupole moment
    dQe1xx = 3 * 2 * X4d * Jx - 2 * rJ
    dQe1xy = 3 * (Y4d * Jx + X4d * Jy)
    dQe1xz = 3 * (Z4d * Jx + X4d * Jz)
    dQe1yy = 3 * 2 * Y4d * Jy - 2 * rJ
    dQe1yz = 3 * (Z4d * Jy + Y4d * Jz)
    dQe1zz = 3 * 2 * Z4d * Jz - 2 * rJ

    dQe2xx = 4 * X4d**2 * rJ - 5 * rr * 2 * X4d * Jx + 2 * rr * rJ
    dQe2xy = 4 * X4d * Y4d * rJ - 5 * rr * (X4d * Jy + Y4d * Jx)
    dQe2xz = 4 * X4d * Z4d * rJ - 5 * rr * (X4d * Jz + Z4d * Jx)
    dQe2yy = 4 * Y4d**2 * rJ - 5 * rr * 2 * Y4d * Jy + 2 * rr * rJ
    dQe2yz = 4 * Y4d * Z4d * rJ - 5 * rr * (Y4d * Jz + Z4d * Jy)
    dQe2zz = 4 * Z4d**2 * rJ - 5 * rr * 2 * Z4d * Jz + 2 * rr * rJ

    Qexx = -1 / (1j * omega) * (trapz4Dto1D(dQe1xx, x, y, z) + k**2 / 14 * trapz4Dto1D(dQe2xx, x, y, z))
    Qexy = -1 / (1j * omega) * (trapz4Dto1D(dQe1xy, x, y, z) + k**2 / 14 * trapz4Dto1D(dQe2xy, x, y, z))
    Qexz = -1 / (1j * omega) * (trapz4Dto1D(dQe1xz, x, y, z) + k**2 / 14 * trapz4Dto1D(dQe2xz, x, y, z))
    Qeyy = -1 / (1j * omega) * (trapz4Dto1D(dQe1yy, x, y, z) + k**2 / 14 * trapz4Dto1D(dQe2yy, x, y, z))
    Qeyz = -1 / (1j * omega) * (trapz4Dto1D(dQe1yz, x, y, z) + k**2 / 14 * trapz4Dto1D(dQe2yz, x, y, z))
    Qezz = -1 / (1j * omega) * (trapz4Dto1D(dQe1zz, x, y, z) + k**2 / 14 * trapz4Dto1D(dQe2zz, x, y, z))

    norm2_Qe = (
        Qexx * np.conj(Qexx) + Qexy * np.conj(Qexy) + Qexz * np.conj(Qexz)
        + Qeyy * np.conj(Qeyy) + Qeyz * np.conj(Qeyz) + Qezz * np.conj(Qezz)
    )
    Cqe = const / 120 * k**2 * norm2_Qe

    ### Magnetic quadrupole moment
    dQmxx = 2 * X4d * rxJx
    dQmxy = X4d * rxJy + Y4d * rxJx
    dQmxz = X4d * rxJz + Z4d * rxJx
    dQmyy = 2 * Y4d * rxJy
    dQmyz = Y4d * rxJz + Z4d * rxJy
    dQmzz = 2 * Z4d * rxJz

    Qmxx = trapz4Dto1D(dQmxx, x, y, z)
    Qmxy = trapz4Dto1D(dQmxy, x, y, z)
    Qmxz = trapz4Dto1D(dQmxz, x, y, z)
    Qmyy = trapz4Dto1D(dQmyy, x, y, z)
    Qmyz = trapz4Dto1D(dQmyz, x, y, z)
    Qmzz = trapz4Dto1D(dQmzz, x, y, z)

    norm2_Qm = (
        Qmxx * np.conj(Qmxx) + Qmxy * np.conj(Qmxy) + Qmxz * np.conj(Qmxz)
        + Qmyy * np.conj(Qmyy) + Qmyz * np.conj(Qmyz) + Qmzz * np.conj(Qmzz)
    )
    Cqm = const / 120 * (k / c)**2 * norm2_Qm

    ### Total cross-section
    Csum = Cp + Cm + Cqe + Cqm

    return Cp,CpT,Cm,Cqe,Cqm,Csum


# Funtion call for calculation of above:
Cp, CpT, Cm, Cqe, Cqm, Csum = calculate_moments(Jx, Jy, Jz, x, y, z, omega, k)


# Output shapes of results (JUST to check)
print(f"Results: Cp={Cp.shape}, Cm={Cm.shape}, Csum={Csum.shape}")


# PLOTTING DATA

import matplotlib.pyplot as plt
import numpy as np

# Assuming Cp, Cm, Csum, and f are numpy arrays with shape (1, 31)
# Flatten them to make them 1D for plotting
f_in_Hz = f_vals.flatten()  # Frequency in Hz
wavelength = (3e8 / f_in_Hz) * 1e9  # Convert wavelength to nanometers

# Ensure that all values are real for plotting
Cp = np.real(Cp.flatten())
CpT = np.real(CpT.flatten())
Cm = np.real(Cm.flatten())
Cqe = np.real(Cqe.flatten())
Cqm = np.real(Cqm.flatten())
Csum = np.real(Csum.flatten())

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(wavelength, Cp, label="Cp", marker='o')
plt.plot(wavelength, CpT, label="CpT", marker='s')
plt.plot(wavelength, Cm, label="Cm", marker='^')
plt.plot(wavelength, Cqe, label="Cqe", marker='v')
plt.plot(wavelength, Cqm, label="Cqm", marker='x')
plt.plot(wavelength, Csum, label="Csum", marker='*')

plt.xlabel("Wavelength (m)")
plt.ylabel("Calculated Values")
plt.title("Moments vs Wavelength")
plt.legend()
plt.grid(True)
plt.show()
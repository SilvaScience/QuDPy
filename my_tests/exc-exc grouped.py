# -*- coding: utf-8 -*-
"""
Created on Thu May 29 13:2x:xx 2026

@author: felix
"""

from itertools import product

from qutip import *
import numpy as np
from qudpy.Classes import *
import qudpy.plot_functions as pf
import ufss
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d


hbar = 0.658211951  # in eV fs
kB = 8.617333262 * 1e-5  # Boltzmann constant eV/K
T = 300  # temperature in K
kT = T * kB
beta = 1 / kT


def spectrum_var(order=3, E_cav=[1.0, 1.1], E=1.1, g=0.05, muc=1.0, muz=1.0,
                 kappa=0.05, gamma_phase=0.15, gamma_decay=0.15, M=2, modes=1, N=1,
                 model="rw", n_th=0.25, decay="no_atomic", T2=None, directory="results",
                 anim="no"):    #choose order
    """
        Plot multiple spectra with real, imaginary and abs values
        :param order: linear(1) or 3rd(3). Must be int data type
        :param E_cav: list of cavity resonant energies
        :param E: excitonic resonant energy
        :param g: # initial coupling strength for cavity mode-spin interaction
        :param muc: dipole strength for the cavity
        :param muz: dipole strength for the particle
        :param kappa: cavity decay strength
        :param gamma_phase: atomic dephasing strength
        :param gamma_decay: atomic decay strength
        :param M: # number of fock basis for cavity mode. Use larger value for stronger couplings
        :param N: # number of lattice sites
        :param model: "no_rw" for no rotating wave approximation, "rw" for rotating wave approximation
        :param n_th: average number thermal photons in the bath coupling to the resonator
        :param decay: "no_atomic" "atomic"
        :param T2: 2nd order time delay
        :param directory: result directory
        :param anim: animate
        :return: Does not return anything unless anim
        """

    if len(E_cav) != modes and modes != 1:
        print("Cavity modes and cavity energy dimensions don't match.")
        return None

    if T2 is None:
        T2 = 10
    if g is None:
        g = 0.05

    if modes == 1:
        if isinstance(E_cav, list):
            wc = E_cav[0]/hbar # cavity resonant frequency
        else: wc = E_cav/hbar
        wz1 = E/hbar # atom resonant frequency
        wz2 = E/hbar * 1.1
        gc = np.sqrt((wz1 + wz2) * wc / 2)/2  # critical coupling strength

        s = N / 2  # Total J for spins.
        n = N + 1  # dimensionality of total spin operator.

        a = tensor(destroy(M), qeye(n), qeye(n))
        Sp1 = tensor(qeye(M), -jmat(s, '+'), qeye(n))
        Sm1 = tensor(qeye(M), -jmat(s, '-'), qeye(n))
        Sx1 = tensor(qeye(M), -jmat(s, 'x'), qeye(n))
        Sz1 = tensor(qeye(M), -jmat(s, 'z'), qeye(n))
        Sp2 = tensor(qeye(M), qeye(n), -jmat(s, '+'))
        Sm2 = tensor(qeye(M), qeye(n), -jmat(s, '-'))
        Sx2 = tensor(qeye(M), qeye(n), -jmat(s, 'x'))
        Sz2 = tensor(qeye(M), qeye(n), -jmat(s, 'z'))
        mud = muc * (a + a.dag()) + muz * Sx1 + muz * Sx2
        ad = a + Sm1 + Sm2

        H_cav = hbar * wc * a.dag() * a # cavity term
        H_exc = hbar * (wz1 * Sz1 + wz2 * Sz2)

        H_exc_exc = 0 # exciton-exciton interaction term

        if model == "no_rw":
            H_int = hbar * (a + a.dag()) * (Sx1 + Sx2)/np.sqrt(2*N)  # intra-cavity interaction term
        elif model == "rw":
            H_int = hbar * (a * (Sp1 + Sp2) + a.dag() * (Sm1 + Sm2))/np.sqrt(2*N)

        H0 = H_cav + H_exc
        H = H0 + g * H_int + H_exc_exc # total hamiltonian
        # print("H0", H0)
        # print("H_int", H_int)
        # print("H", H)

        # collapse operators: cavity relaxation, cavity exc., collective dephasing, atomic relaxation, atomic exc.
        c_cav_rel = np.sqrt(kappa * (n_th + 1)) * a
        c_cav_exc = np.sqrt(kappa * n_th) * a.dag()
        c_col_dep = np.sqrt(gamma_phase) * (Sz1 + Sz2) / np.sqrt(2)
        c_ato_rel = np.sqrt(gamma_decay * (n_th + 1)) * (Sm1 + Sm2) / np.sqrt(2*N)
        c_ato_exc =np.sqrt(gamma_decay * n_th) * (Sp1 + Sp2) / np.sqrt(2*N)
        c_ops = [c_cav_rel, c_cav_exc, c_col_dep]

    if modes == 0:
        if isinstance(E_cav, list):
            wc = E_cav[0]/hbar # cavity resonant frequency
        else: wc = E_cav/hbar
        wz1 = E/hbar # atom resonant frequency
        wz2 = E/hbar * 1.1
        gc = np.sqrt((wz1 + wz2) * wc / 2)/2  # critical coupling strength

        s = N / 2  # Total J for spins.
        n = N + 1  # dimensionality of total spin operator.

        a = tensor(destroy(M), qeye(n), qeye(n))
        Sp1 = tensor(qeye(M), -jmat(s, '+'), qeye(n))
        Sm1 = tensor(qeye(M), -jmat(s, '-'), qeye(n))
        Sx1 = tensor(qeye(M), -jmat(s, 'x'), qeye(n))
        Sz1 = tensor(qeye(M), -jmat(s, 'z'), qeye(n))
        mud = muc * (a + a.dag()) + muz * Sx1 + muz * Sx2
        ad = a + Sm1 + Sm2

        H_cav = hbar * wc * a.dag() * a # cavity term
        H_exc = hbar * (wz1 * Sz1 + wz2 * Sz2)

        H_exc_exc = 0 # exciton-exciton interaction term

        if model == "no_rw":
            H_int = hbar * (a + a.dag()) * (Sx1 + Sx2)/np.sqrt(2*N)  # intra-cavity interaction term
        elif model == "rw":
            H_int = hbar * (a * (Sp1 + Sp2) + a.dag() * (Sm1 + Sm2))/np.sqrt(2*N)

        H0 = H_cav + H_exc
        H = H0 + g * H_int + H_exc_exc # total hamiltonian
        # print("H0", H0)
        # print("H_int", H_int)
        # print("H", H)

        # collapse operators: cavity relaxation, cavity exc., collective dephasing, atomic relaxation, atomic exc.
        c_cav_rel = np.sqrt(kappa * (n_th + 1)) * a
        c_cav_exc = np.sqrt(kappa * n_th) * a.dag()
        c_col_dep = np.sqrt(gamma_phase) * (Sz1 + Sz2) / np.sqrt(2)
        c_ato_rel = np.sqrt(gamma_decay * (n_th + 1)) * (Sm1 + Sm2) / np.sqrt(2*N)
        c_ato_exc =np.sqrt(gamma_decay * n_th) * (Sp1 + Sp2) / np.sqrt(2*N)
        c_ops = [c_cav_rel, c_cav_exc, c_col_dep]

    title = "exc_exc_model"

    # print("dimensionality of Hilbert-space: ", H.shape)

    # setting up system
    rho = tensor([fock_dm(M, 0)] + [fock_dm(N+1, 0)] + [fock_dm(N+1, 0)])  # ground state of Hamiltonian
    sys = System(H=H, rho=rho, a=ad, u=mud, c_ops=c_ops, diagonalize=True)

    en, T = H.eigenstates()

    # print("system has been intialized")

    # # calculate the expectation value of the number of photons in the cavity
    # n_vec = expect(a.dag() * a, rho)
    # n2_vec = expect(a.dag() * a*a.dag() * a, rho)
    # Jz_vec = expect(Sz, rho)
    # a_vec = expect(a, rho)

    if order == 1:
        d0 = 100

        # generating spectra
        dipole, t_list, spec, freq = sys.linear_spec(d0, r=1.5/np.pi, title_graph=title, plot_graph=False, dir=directory)

    if order == 3:

        # Setting up the required double sided diagrams for tests
        # DiagramGenerator class, or DG for short
        DG = ufss.DiagramGenerator
        # initialize the module

        R3rd = DG()  # DG takes a single key-word argument, which has the default value detection_type = 'polarization'
        # DiagramAutomation needs to know the phase-matching/-cycling condition
        R3rd.set_phase_discrimination([(0, 1), (1, 0), (1, 0)])  # setting phase-matching condition for rephasing diagrams R1,2,3
        # Set the pulse interval
        t_pulse = np.array([-1, 1])
        R3rd.efield_times = [t_pulse] * 4

        # Creating diagrams for pulse arrival times 0, 100, 200 and detection time 300.
        [R3, R1, R2] = R3rd.get_diagrams([0, 100, 200, 300])
        rephasing = [R1, R2, R3]
        # print('the rephasing diagrams are R1, R2 and R3 ', rephasing)

        # setting conditions for and generating non-rephasing diagrams R4, R5 and R6
        R3rd.set_phase_discrimination([(1, 0), (0, 1), (1, 0)])
        [R6, R4, R5] = R3rd.get_diagrams([0, 100, 200, 200])
        nonrephasing = [R4, R5, R6]
        # print('the non-rephasing diagrams are R4, R5 and R6', nonrephasing)

        sys.diagram_donkey([0, 100, 100+T2, 200+T2], [R3], r=10, title_graph=None, plot_graph=False, dir=directory)

        # print("finished setup stage")

        # generating 2Dcoherence response for rephasing diagrams
        time_delays = [100, T2, 100]
        scan_id = [0, 2]
        response_list = []
        diagrams = rephasing + nonrephasing
        for k in range(6):
            states, t1, t2, dipole = sys.coherence2d(time_delays, diagrams[k], scan_id, r=1.5/np.pi, parallel=True)
            # print('diagram ', k, ' done')
            response_list.append(1j * dipole)
        spectra_list, extent, f1, f2 = sys.spectra(np.imag(response_list), resolution=1)

        rephasing_spectra = spectra_list[:3]
        rephasing_spectra.append(np.sum(spectra_list[:3], 0))
        pf.silva_plot_contourf(rephasing_spectra, f1, f2, labels=['E emission', 'E absorption'],
                  scale='linear', color_map='PuOr', title_list = ['$R_1$', '$R_2$', '$R_3$', '$R_{rephasing}$'],
                  center_scale=False, plot_sum=False, plot_quadrant='2', zoom_coor=[-2.8,-2,2,2.8],
                  invert_y=False, diagonals=[True, False], nlevels=10,
                  title_graph=None, plot_graph=True, dir=directory)

spectrum_var()
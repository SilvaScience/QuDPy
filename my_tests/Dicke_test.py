# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:16:44 2026

@author: felix
"""

from itertools import product

from qutip import *
import numpy as np
from qudpy.Classes import *
import qudpy.plot_functions as pf
import ufss

#choose order
order = 3

hbar = 0.658211951  # in eV fs
E0 = 1.
E = 1.1
wc = E0/hbar
wz = E/hbar
g = 0.01  # initial coupling strength for cavity mode-spin interaction for demonstration purposes only
gc = np.sqrt(wz * wc)/2  # critical coupling strength
print("critical coupling strength = ", gc)
mu_str = 1.

kB = 8.617333262*1e-5  # Boltzmann constant eV/K
T = 300  # temperature in K
kT = T*kB
beta = 1/kT
kappa = 0.05
gamma_phase = 0.15
gamma_decay = 0.15
M = 2   # number of fock basis for cavity mode. Use larger value for stronger couplings
N = 2    # number of spins
s = N/2  # Total J for spins.
n = N+1  # dimensionality of total spin operator.


a = tensor(destroy(M), qeye(n))
Sp = tensor(qeye(M), -jmat(s, '+'))
Sm = tensor(qeye(M), -jmat(s, '-'))
Sx = tensor(qeye(M), -jmat(s, 'x'))
Sy = tensor(qeye(M), -jmat(s, 'y'))
Sz = tensor(qeye(M), -jmat(s, 'z'))
mud = mu_str * (a + a.dag()) + mu_str * Sx
ad = a + Sm


H0 = hbar * (wc * a.dag() * a + wz * Sz)    # default basic Hamiltonian
H1_no_rw = hbar * (a + a.dag()) * Sx/np.sqrt(N)  # intra-cavity interaction term (no rotating wave approx.)
H1_rw = hbar * (a * Sp + a.dag() * Sm)/np.sqrt(N)  # intra-cavity interaction term (with rotating wave approx.)
H_no_rw = H0 + g * H1_no_rw
H_rw = H0 + g * H1_rw

# average number thermal photons in the bath coupling to the resonator
n_th = 0.25
# collapse operators: cavity relaxation, cavity exc., collective dephasing, atomic relaxation, atomic exc.
c_cav_rel = np.sqrt(kappa * (n_th + 1)) * a
c_cav_exc = np.sqrt(kappa * n_th) * a.dag()
c_col_dep = np.sqrt(gamma_phase)*Sz
c_ato_rel = np.sqrt(gamma_decay * (n_th + 1)) * Sm
c_ato_exc =np.sqrt(gamma_decay * n_th) * Sp

c_ops_base = [c_cav_rel, c_cav_exc, c_col_dep]
c_ops_more = [c_cav_rel, c_cav_exc, c_col_dep, c_ato_rel, c_ato_exc]



H_c_ops = list(product([H_no_rw, H_rw], [c_ops_base, c_ops_more]))
titles = ["no rw, base collapse", "no rw, atomic collapse", "rw, base collapse", "rw, atomic collapse"]

for (H, c_ops), title in zip(H_c_ops, titles):
    title = "M"+str(M)+" N"+str(N)+" order"+str(order)+" "+title
    print(title+"\n")

    print("dimensionality of Hilbert-space: ", H.shape)

    # setting up system
    rho = tensor(fock_dm(M, 0), fock_dm(N+1, 0))  # ground state of Hamiltonian
    sys = System(H=H, rho=rho, a=ad, u=mud, c_ops=c_ops, diagonalize=True)

    print("system has been intialized")

    # calculate the expectation value of the number of photons in the cavity
    n_vec = expect(a.dag() * a, rho)
    n2_vec = expect(a.dag() * a*a.dag() * a, rho)
    Jz_vec = expect(Sz, rho)
    a_vec = expect(a, rho)

    if order == 1:
        d0 = 100

        # generating spectra
        dipole, t_list, spec, freq = sys.linear_spec(d0, r=1.5/np.pi, title_graph=title, plot_graph=False, dir="dicke_results")

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
        [R3, R1, R2] = R3rd.get_diagrams([0, 100, 110, 210])
        rephasing = [R1, R2, R3]
        # print('the rephasing diagrams are R1, R2 and R3 ', rephasing)

        # setting conditions for and generating non-rephasing diagrams R4, R5 and R6
        R3rd.set_phase_discrimination([(1, 0), (0, 1), (1, 0)])
        [R6, R4, R5] = R3rd.get_diagrams([0, 100, 110, 210])
        nonrephasing = [R4, R5, R6]
        # print('the non-rephasing diagrams are R4, R5 and R6', nonrephasing)

        states = sys.diagram_donkey([0, 100, 110, 210], [R1], r=10, title_graph=title, plot_graph=False, dir="dicke_results")

        print("finished setup stage")

        # generating 2Dcoherence response for rephasing diagrams
        time_delays = [100, 10, 100]
        scan_id = [0, 2]
        response_list = []
        states_list = []
        diagrams = rephasing + nonrephasing
        for k in range(6):
            states, t1, t2, dipole = sys.coherence2d(time_delays, diagrams[k], scan_id, r=1.5/np.pi, parallel=False)
            print('diagram ', k, ' done')
            response_list.append(1j * dipole)
            states_list.append(states)
        spectra_list, extent, f1, f2 = sys.spectra(np.imag(response_list))

        rephasing_spectra = spectra_list[:3]
        rephasing_spectra.append(np.sum(spectra_list[:3], 0))
        pf.silva_plot_contourf(rephasing_spectra, f1, f2, labels=['E emission', 'E absorption'],
                      scale='linear', color_map='PuOr', center_scale=False, plot_sum=False, plot_quadrant='2', invert_y=False,
                      diagonals=[True, False], title_graph=title, plot_graph=False, direc="dicke_results/")
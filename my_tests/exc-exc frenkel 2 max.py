# -*- coding: utf-8 -*-
"""
Created on Thu June 2 14:3x:xx 2026

@author: felix
"""

from itertools import product
import sys

from qutip import *
import numpy as np
from qudpy.Classes import *
import qudpy.plot_functions as pf
import ufss
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import copy

np.set_printoptions(threshold=sys.maxsize)


def to_ternary(n: int) -> str: # used AI for this ngl, saves time
    """Converts an integer to its standard base-3 (ternary) string representation."""
    if n == 0:
        return "0"
    digits = []
    while n > 0:
        # Append the remainder of division by 3
        digits.append(str(n % 3))
        # Keep the quotient for the next loop pass
        n //= 3
    # Reverse the digits and join them
    ternary_str = "".join(reversed(digits))
    return ternary_str

hbar = 0.658211951  # in eV fs
kB = 8.617333262 * 1e-5  # Boltzmann constant eV/K
T = 300  # temperature in K
kT = T * kB
beta = 1 / kT

def spectrum_var(order=3, en_cav=None, en_exc=None, g=0.05, muc=1.0, muz=1.0,
                 kappa=0.1, gamma_phase=0.15, gamma_decay=0.15, hh_bind=0.02,
                 ll_bind=0.05, hl_bind=0.01, hg_swap=0.05, lg_swap=0.04, M=2, modes=2, N=2, model="no_rw",
                 n_th=0.25, decay="no_atomic", time2=None, directory="results", anim="no"):
    """
        Plot multiple spectra with real, imaginary and abs values
        :param order: linear(1) or 3rd(3). Must be int data type
        :param en_cav: list of cavity resonant energies
        :param en_exc: excitonic resonant energy
        :param g: # initial coupling strength for cavity mode-spin interaction
        :param muc: dipole strength for the cavity
        :param muz: dipole strength for the particle
        :param kappa: cavity decay strength
        :param gamma_phase: atomic dephasing strength
        :param gamma_decay: atomic decay strength
        :param hh_bind: strength of neighboring HH binding
        :param ll_bind: strength of neighboring LL binding
        :param hl_bind: strength of neighboring HL/LH binding
        :param hg_swap: strength of neighboring H-ground swapping
        :param lg_swap: strength of neighboring L-ground swapping
        :param M: # number of fock basis for cavity mode. Use larger value for stronger couplings
        :param N: # number of lattice sites
        :param model: "no_rw" for no rotating wave approximation, "rw" for rotating wave approximation
        :param n_th: average number thermal photons in the bath coupling to the resonator
        :param decay: "no_atomic" "atomic"
        :param time2: 2nd order time delay
        :param directory: result directory
        :param anim: animate
        :return: Does not return anything unless anim
        """
    if en_cav is None:
        if modes == 1: en_cav = 1.0
        else:
            en_cav = [1.0, 1.1]
            en_cav = np.array(en_cav)
    if en_exc is None:
        en_exc = [1.0, 1.1]
    en_exc = np.array(en_exc)
    if modes != 1:
        if len(en_cav) != modes:
            print("Cavity modes and cavity energy dimensions don't match.")
            return None

    if modes == 1:

        # setting up dm
        fock_ = fock_dm(M, 0)
        exc_dims = "0" * N
        exc_basis = ket(exc_dims, 3)
        exciton_space = exc_basis * exc_basis.dag()  # ground state of Hamiltonian
        rho = tensor(fock_, exciton_space)
        null = np.zeros((3 ** N, 3 ** N))

        if isinstance(en_cav, list):
            wc = en_cav[0]/hbar # cavity resonant frequency
        else: wc = en_cav/hbar
        wz1, wz2 = en_exc / hbar  # atom resonant frequencies

        a = tensor([destroy(M)] + [qeye(3) for _ in range(N)])

        H_cav = hbar * (wc * a.dag() * a)  # cavity term

        H_exc_diag_sep = copy.deepcopy(null)
        H_exc_binding_sep = copy.deepcopy(null)
        H_exc_swap_sep = copy.deepcopy(null)
        sites = [[[] for _ in range(3)] for _ in range(
            3 ** N)]  # sites has 3**N lists of 3 lists. These 3 lists have 2 items among them, the 2 positions of excitons
        site_state = [[[] for _ in range(3)] for _ in range(
            N)]  # [i,j]; i is the site index, j is {0,H,L}. int is the state number in reduced hilbert space
        idx_tern_list = []
        wann_h = [[[], []], [[], []]]
        wann_l = [[[], []], [[], []]]

        for i in range(3 ** N):  # diagonal eigen energy is just freq of H-exc * nb of H-exc + idem for L-exc.
            # in ternary, you get the # of '1' and '2' which are the nb of each exciton. Viva les 3-lvl systems
            # same for neighboring HH/LL/HL/LH, find "11"/"22"/"12"/"21" then binding energy
            idx_tern = str(to_ternary(i))  # string of index in ternary
            idx_tern_list.append(idx_tern)
            if i < 3 ** N: idx_tern = '0' * (N - len(
                idx_tern)) + idx_tern  # make sure it has at least N terms. I think this works #,# need to test with N>2 but cba rn

            en_eigen = wz1 * idx_tern.count('1') + wz2 * idx_tern.count('2')
            H_exc_diag_sep[i, i] = en_eigen

            en_binding = (hh_bind * idx_tern.count("11") + ll_bind * idx_tern.count("22") +
                          hl_bind * (idx_tern.count("12") + idx_tern.count("21")))
            H_exc_binding_sep[i, i] = -en_binding

            if idx_tern.count("01"):
                wann_h[0][0].append(i)
                wann_h[0][1].append(idx_tern.find("01"))
            if idx_tern.count("10"):
                wann_h[1][0].append(i)
                wann_h[1][1].append(idx_tern.find("10"))
            if idx_tern.count("02"):
                wann_l[0][0].append(i)
                wann_l[0][1].append(idx_tern.find("02"))
            if idx_tern.count("20"):
                wann_l[1][0].append(i)
                wann_l[1][1].append(idx_tern.find("20"))

            # sites = [[] for _ in range(3)]
            for k in range(len(idx_tern)):  # should be same as range(N) if the if i < 3**N works fine
                sites[i][int(idx_tern[k])].append(k)
                site_state[k][int(idx_tern[k])].append(i)

        h_ground_sep = [copy.deepcopy(null) for _ in range(N)]
        l_ground_sep = [copy.deepcopy(null) for _ in range(N)]
        for i in range(N):
            h_ground_sep[i][site_state[N - 1 - i][0], site_state[N - 1 - i][1]] = 1
            l_ground_sep[i][site_state[N - 1 - i][0], site_state[N - 1 - i][2]] = 1
        h_ground = [tensor([qeye(M)] + [Qobj(h_ground_sep[k], dims=[[3 for _ in range(N)] for _ in range(2)])])
                    for k in range(N)]  # lowering for H, index is exciton index from right to left
        l_ground = [tensor([qeye(M)] + [Qobj(l_ground_sep[k], dims=[[3 for _ in range(N)] for _ in range(2)])])
                    for k in range(N)]  # idem for L

        for j in range(len(wann_h[0][0])):
            for k in range(len(wann_h[0][0])):
                if wann_h[0][1][j] == wann_h[1][1][k]:
                    H_exc_swap_sep[wann_h[0][0][j]][wann_h[1][0][k]] = hg_swap
                if wann_l[0][1][j] == wann_l[1][1][k]:
                    H_exc_swap_sep[wann_l[0][0][j]][wann_l[1][0][k]] = lg_swap

        H_exc = tensor([qeye(M)] + [Qobj(H_exc_diag_sep,
                                dims=[[3 for _ in range(N)] for _ in range(2)])]) # exciton diag pop terms

        H_exc_frenkel = tensor([qeye(M)] + [Qobj(H_exc_binding_sep,
                                dims=[[3 for _ in range(N)] for _ in range(2)])])  # frenkel neighboring binding term
        H_exc_wanmott = tensor([qeye(M)] + [Qobj(H_exc_swap_sep,
                                dims=[[3 for _ in range(N)] for _ in range(2)])])  # wannier-mott swapping with neighboring ground state term

        H_exc_exc = H_exc_frenkel + H_exc_wanmott  # exciton-exciton interaction term

        if model == "no_rw":
            H_int = hbar * g * ((a + a.dag()) * (sum(h_ground) + sum([h_ground[k].dag() for k in range(N)]) +
                                                 sum(l_ground) + sum([l_ground[k].dag() for k in range(N)])))
        elif model == "rw":
            H_int = hbar * g * (a * (sum([h_ground[k].dag() for k in range(N)]) + sum([l_ground[k].dag()
                                    for k in range(N)])) + a.dag() * (h_ground + l_ground))

        H0 = H_cav + H_exc
        H = H0 + g * H_int + H_exc_exc  # total hamiltonian
        # print("H0", H0)
        # print("H_int", H_int)
        # print("H", H)

        mud = muc * (a + a.dag()) + muz * (sum(h_ground) + sum([h_ground[k].dag() for k in range(N)]) +
                    sum(l_ground) + sum([l_ground[k].dag() for k in range(N)]))  # sqrt(N)? sqrt(3**N)? sqrt(perms)?
        ad = muc * a + muz * (sum(h_ground) + sum(l_ground))

        # collapse operators: cavity relaxation, cavity exc., collective dephasing, atomic relaxation, atomic exc.
        c_cav_rel = np.sqrt(kappa * (n_th + 1)) * a
        c_cav_exc = np.sqrt(kappa * n_th) * a.dag()
        c_col_dep = np.sqrt(gamma_phase) * H_exc
        c_ato_rel = np.sqrt(gamma_decay * (n_th + 1)) * (sum(h_ground) + sum(l_ground))
        c_ato_exc = np.sqrt(gamma_decay * n_th) * (
                    sum([h_ground[k].dag() for k in range(N)]) + sum([l_ground[k].dag() for k in range(N)]))
        c_ops = [c_cav_rel, c_cav_exc, c_col_dep]

    elif modes == 2:

        # setting up dm
        fock1 = fock_dm(M, 0)
        fock2 = fock_dm(M, 0)
        exc_dims = "0" * N
        exc_basis = ket(exc_dims, 3)
        exciton_space = exc_basis * exc_basis.dag()  # ground state of Hamiltonian
        rho = tensor(fock1, fock2, exciton_space)
        null = np.zeros((3 ** N, 3 ** N))

        wc1,wc2 = en_cav / hbar  # cavity resonant frequencies
        wz1, wz2 = en_exc / hbar  # atom resonant frequencies

        a = tensor([destroy(M)] + [qeye(M)] + [qeye(3) for _ in range(N)])
        b = tensor([qeye(M)] + [destroy(M)] + [qeye(3) for _ in range(N)])

        H_cav = hbar * (wc1 * a.dag() * a + wc2 * b.dag() * b)  # cavity terms

        H_exc_diag_sep = copy.deepcopy(null)
        H_exc_binding_sep = copy.deepcopy(null)
        H_exc_swap_sep = copy.deepcopy(null)
        sites = [[[] for _ in range(3)] for _ in range(3**N)] # sites has 3**N lists of 3 lists. These 3 lists have 2 items among them, the 2 positions of excitons
        site_state = [[[] for _ in range(3)] for _ in range(N)] # [i,j]; i is the site index, j is {0,H,L}. int is the state number in reduced hilbert space
        idx_tern_list = []
        wann_h = [[[],[]],[[],[]]]
        wann_l = [[[],[]],[[],[]]]

        for i in range(3**N): # diagonal eigen energy is just freq of H-exc * nb of H-exc + idem for L-exc.
                    #in ternary, you get the # of '1' and '2' which are the nb of each exciton. Viva les 3-lvl systems
                    #same for neighboring HH/LL/HL/LH, find "11"/"22"/"12"/"21" then binding energy
            idx_tern = str(to_ternary(i)) # string of index in ternary
            idx_tern_list.append(idx_tern)
            if i < 3**N: idx_tern = '0'*(N-len(idx_tern)) + idx_tern # make sure it has at least N terms. I think this works #,# need to test with N>2 but cba rn

            en_eigen = wz1 * idx_tern.count('1') + wz2 * idx_tern.count('2')
            H_exc_diag_sep[i, i] = en_eigen

            en_binding = (hh_bind * idx_tern.count("11") + ll_bind * idx_tern.count("22") +
                          hl_bind * (idx_tern.count("12") + idx_tern.count("21")))
            H_exc_binding_sep[i,i] = -en_binding

            if idx_tern.count("01"):
                wann_h[0][0].append(i)
                wann_h[0][1].append(idx_tern.find("01"))
            if idx_tern.count("10"):
                wann_h[1][0].append(i)
                wann_h[1][1].append(idx_tern.find("10"))
            if idx_tern.count("02"):
                wann_l[0][0].append(i)
                wann_l[0][1].append(idx_tern.find("02"))
            if idx_tern.count("20"):
                wann_l[1][0].append(i)
                wann_l[1][1].append(idx_tern.find("20"))

            # sites = [[] for _ in range(3)]
            for k in range(len(idx_tern)): # should be same as range(N) if the if i < 3**N works fine
                sites[i][int(idx_tern[k])].append(k)
                site_state[k][int(idx_tern[k])].append(i)

        h_ground_sep = [copy.deepcopy(null) for _ in range(N)]
        l_ground_sep = [copy.deepcopy(null) for _ in range(N)]
        for i in range(N):
            h_ground_sep[i][site_state[N-1-i][0], site_state[N-1-i][1]] = 1
            l_ground_sep[i][site_state[N-1-i][0], site_state[N-1-i][2]] = 1
        h_ground = [tensor([qeye(M), qeye(M)] + [Qobj(h_ground_sep[k], dims=[[3 for _ in range(N)] for _ in range(2)])]) for k in range(N)] # lowering for H, index is exciton index from right to left
        l_ground = [tensor([qeye(M), qeye(M)] + [Qobj(l_ground_sep[k], dims=[[3 for _ in range(N)] for _ in range(2)])]) for k in range(N)] # idem for L

        for j in range(len(wann_h[0][0])):
            for k in range(len(wann_h[0][0])):
                if wann_h[0][1][j] == wann_h[1][1][k]:
                    H_exc_swap_sep[wann_h[0][0][j]][wann_h[1][0][k]] = hg_swap
                if wann_l[0][1][j] == wann_l[1][1][k]:
                    H_exc_swap_sep[wann_l[0][0][j]][wann_l[1][0][k]] = lg_swap

        H_exc = tensor([qeye(M), qeye(M)] + [Qobj(H_exc_diag_sep, dims=[[3 for _ in range(N)] for _ in range(2)])])  # exciton diag pop terms

        H_exc_frenkel = tensor([qeye(M), qeye(M)] + [Qobj(H_exc_binding_sep,
                                        dims=[[3 for _ in range(N)] for _ in range(2)])]) # frenkel neighboring binding term
        H_exc_wanmott = tensor([qeye(M), qeye(M)] + [Qobj(H_exc_swap_sep,
                                        dims=[[3 for _ in range(N)] for _ in range(2)])]) # wannier-mott swapping with neighboring ground state term

        H_exc_exc = H_exc_frenkel + H_exc_wanmott  # exciton-exciton interaction term

        if model == "no_rw":
            H_int = hbar * g * ((a + a.dag()) * (sum(h_ground) + sum([h_ground[k].dag() for k in range(N)])) +
                                (b + b.dag()) * (sum(l_ground) + sum([l_ground[k].dag() for k in range(N)])))
        elif model == "rw":
            H_int = hbar * g * (a * sum([h_ground[k].dag() for k in range(N)]) + a.dag() * h_ground +
                                b * sum([l_ground[k].dag() for k in range(N)]) + b.dag() * l_ground)

        H0 = H_cav + H_exc
        H = H0 + g * H_int + H_exc_exc # total hamiltonian
        # print("H0", H0)
        # print("H_int", H_int)
        # print("H", H)

        mud = muc * (a + a.dag() + b + b.dag()) + muz * (sum(h_ground) + sum([h_ground[k].dag() for k in range(N)]) +
                                                         sum(l_ground) + sum([l_ground[k].dag() for k in range(N)])) # sqrt(N)? sqrt(3**N)? sqrt(perms)?
        ad = muc * (a + b) + muz * (sum(h_ground) + sum(l_ground))

        # collapse operators: cavity relaxation, cavity exc., collective dephasing, atomic relaxation, atomic exc.
        c_cav1_rel = np.sqrt(kappa * (n_th + 1)) * a
        c_cav2_rel = np.sqrt(kappa * (n_th + 1)) * b
        c_cav1_exc = np.sqrt(kappa * n_th) * a.dag()
        c_cav2_exc = np.sqrt(kappa * n_th) * b.dag()
        c_col_dep = np.sqrt(gamma_phase) * H_exc
        c_ato_rel = np.sqrt(gamma_decay * (n_th + 1)) * (sum(h_ground) + sum(l_ground))
        c_ato_exc =np.sqrt(gamma_decay * n_th) * (sum([h_ground[k].dag() for k in range(N)]) + sum([l_ground[k].dag() for k in range(N)]))
        c_ops = [c_cav1_rel, c_cav2_rel, c_cav1_exc, c_cav2_exc, c_col_dep]

    if time2 is None:
        time2 = 10
    if g is None:
        g = 0.05

    title = "exc_exc_model"

    print("dimensionality of Hilbert-space: ", H.shape)

    # setting up system
    sys = System(H=H, rho=rho, a=ad, u=mud, c_ops=c_ops, diagonalize=True)

    en, T = H.eigenstates()

    print("system has been intialized")

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
        print('the rephasing diagrams are R1, R2 and R3 ', rephasing)

        # setting conditions for and generating non-rephasing diagrams R4, R5 and R6
        R3rd.set_phase_discrimination([(1, 0), (0, 1), (1, 0)])
        [R6, R4, R5] = R3rd.get_diagrams([0, 100, 200, 200])
        nonrephasing = [R4, R5, R6]
        print('the non-rephasing diagrams are R4, R5 and R6', nonrephasing)

        sys.diagram_donkey([0, 100, 100+time2, 200+time2], [R3], r=10, title_graph=None, plot_graph=False, dir=directory)

        print("finished setup stage")

        # generating 2Dcoherence response for rephasing diagrams
        time_delays = [100, time2, 100]
        scan_id = [0, 2]
        response_list = []
        diagrams = rephasing + nonrephasing
        for k in range(6):
            states, t1, t2, dipole = sys.coherence2d(time_delays, diagrams[k], scan_id, r=1.5/np.pi, parallel=True)
            print('diagram ', k, ' done')
            response_list.append(1j * dipole)
        spectra_list, extent, f1, f2 = sys.spectra(np.imag(response_list), resolution=1)

        rephasing_spectra = spectra_list[:3]
        rephasing_spectra.append(np.sum(spectra_list[:3], 0))
        pf.silva_plot_contourf(rephasing_spectra, f1, f2, labels=['E emission', 'E absorption'],
                  scale='linear', color_map='PuOr', title_list = ['$R_1$', '$R_2$', '$R_3$', '$R_{rephasing}$'],
                  center_scale=False, plot_sum=False, plot_quadrant='2', zoom_coor=[-2.8,-2,2,2.8],
                  invert_y=False, diagonals=[True, False], nlevels=10,
                  title_graph=None, plot_graph=True, direc=directory)

spectrum_var(N=2, modes=1)
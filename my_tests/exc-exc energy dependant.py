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

np.set_printoptions(threshold=sys.maxsize, linewidth=190, precision=3, suppress=True)

hbar = 0.658211951  # in eV fs
kB = 8.617333262 * 1e-5  # Boltzmann constant eV/K
T = 300  # temperature in K
kT = T * kB
beta = 1 / kT

def spectrum_var(order=3, en_cav=None, en_exc=None, g=0.01, J=0.01,
                 muc=1.0, muz=1.0, kappa=0.01, gamma_phase=0.01, gamma_decay=0.1,
                 gamma_hl=0.01,
                 hh_bind=0.2, ll_bind=0.2, hl_bind=0.2, hg_swap=0.1, lg_swap=0.1,
                 h_str = 1, l_str = 0.0, M=3, modes=1, dark="k", N=2, model="no_rw", n_th=0.1,
                 decay="no_atomic", time2=None, directory="results/exc-exc/", anim="no"):
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
        en_cav = 1.0
    if en_exc is None:
        en_exc = [1.0, 1.1]
    en_exc = np.array(en_exc)
    if time2 is None:
        time2 = 20
    if g is None:
        g = 0.05

    if N == 2:

        # setting up dm
        fock_ = fock_dm(M, 0)
        exc_basis0 = np.zeros((6, 6));
        exc_basis0[0, 0] = 1;
        exc_basis = Qobj(exc_basis0)  # ground,1h,1l,2h-bound,hl-bound,2lbound
        exciton_space = exc_basis * exc_basis.dag()  # ground state of Hamiltonian
        rho = tensor(fock_, exciton_space)

        wc = en_cav / hbar
        wz_h, wz_l = en_exc / hbar  # atom resonant frequencies

        a = tensor([destroy(M)] + [qeye(6)])

        H_cav = hbar * (wc * a.dag() * a)  # cavity term

        # each individual lowering operators
        h_low = np.zeros_like(exc_basis0);
        h_low[0, 1] = 1;
        h_low = tensor(qeye(M), Qobj(h_low))
        hh_bound_low = np.zeros_like(exc_basis0);
        hh_bound_low[1, 3] = 1;
        hh_bound_low = tensor(qeye(M), Qobj(hh_bound_low))
        l_low = np.zeros_like(exc_basis0);
        l_low[0, 2] = 1;
        l_low = tensor(qeye(M), Qobj(l_low))
        ll_bound_low = np.zeros_like(exc_basis0);
        ll_bound_low[2, 5] = 1;
        ll_bound_low = tensor(qeye(M), Qobj(ll_bound_low))
        hl_h_bound_low = np.zeros_like(exc_basis0);
        hl_h_bound_low[1, 4] = 1 / np.sqrt(2);
        hl_h_bound_low = tensor(qeye(M), Qobj(hl_h_bound_low))  # lowers to h
        hl_l_bound_low = np.zeros_like(exc_basis0);
        hl_l_bound_low[2, 4] = 1 / np.sqrt(2);
        hl_l_bound_low = tensor(qeye(M), Qobj(hl_l_bound_low))  # lowers to l

        exc_h_low = h_low + hh_bound_low + hl_l_bound_low
        exc_l_low = l_low + hl_h_bound_low + ll_bound_low
        if dark == "light": exc_low = exc_h_low
        elif dark == "heavy": exc_low = exc_l_low
        else: exc_low = exc_h_low + exc_l_low

        H_exc = (hbar * wz_h * h_low.dag() * h_low +  # h-pop
                 hbar * wz_l * l_low.dag() * l_low +  # l-pop
                 hbar * (2 * wz_h - hh_bind / 2) * hh_bound_low.dag() * hh_bound_low +  # hh-biexc-pop
                 hbar * (2 * wz_l - ll_bind / 2) * ll_bound_low.dag() * ll_bound_low +  # ll-biexc-pop
                 hbar * (wz_h + wz_l - hl_bind / 2) * (
                     hl_h_bound_low.dag() * hl_h_bound_low + hl_l_bound_low.dag() * hl_l_bound_low))  # hl-biexc-pop

        # only 1 interacts with the cavity
        if model == "no_rw":
            if dark == "light":
                H_int = hbar * (a + a.dag()) * (exc_h_low + exc_h_low.dag())
            elif dark == "heavy":
                H_int = hbar * (a + a.dag()) * (exc_l_low + exc_l_low.dag())
            else:
                H_int = hbar * (a + a.dag()) * (h_str * (exc_h_low + exc_h_low.dag()) + l_str * (exc_l_low + exc_l_low.dag()))
        elif model == "rw":
            if dark == "light":
                H_int = hbar * (a.dag() * exc_h_low + a * exc_h_low.dag())
            elif dark == "heavy":
                H_int = hbar * (a.dag() * exc_l_low + a * exc_l_low.dag())
            else:
                H_int = hbar * (a.dag() * (h_str * exc_h_low + l_str * exc_l_low) +
                                a * (h_str * exc_h_low.dag() + l_str * exc_l_low.dag()))

        H_coupling = h_low.dag() * l_low  # H-L coupling
        H_coupling += hh_bound_low.dag() * hl_h_bound_low * np.sqrt(2) + hl_l_bound_low.dag() * ll_bound_low * np.sqrt(2)
        # print("H_coupling", H_coupling)

        H0 = H_cav + H_exc  # diagonal hamiltonian
        H = H0 + g * H_int + J * (H_coupling + H_coupling.dag())  # total hamiltonian
        # print("H0", H0)
        print("H_int", H_int)
        # print("H", H)

        mud = muc * (a + a.dag()) + muz * (exc_low + exc_low.dag())
        ad = muc * a + muz * exc_low
        # print("mud", mud)
        print("ad", ad)
        print("adag", ad.dag())

        # collapse operators: cavity relaxation, cavity exc., collective dephasing, atomic relaxation, atomic exc.
        c_cav_rel = np.sqrt(kappa * (n_th + 1)) * a
        c_cav_exc = np.sqrt(kappa * n_th) * a.dag()
        c_col_dep = np.sqrt(gamma_phase) * H_exc
        c_ato_rel = np.sqrt(gamma_decay * (n_th + 1)) * exc_low
        c_ato_exc = np.sqrt(gamma_decay * n_th) * exc_low.dag()
        c_h_l = np.sqrt(gamma_hl * (n_th + 1)) * exc_l_low.dag() * exc_h_low
        c_l_h = np.sqrt(gamma_hl * n_th) * exc_h_low.dag() * exc_l_low
        c_ops = [c_cav_rel, c_cav_exc, c_col_dep, c_h_l, c_l_h]

    if N == 3:

        # setting up dm
        fock_ = fock_dm(M, 0)
        exc_basis0 = np.zeros((12, 12));
        exc_basis0[0, 0] = 1;
        exc_basis = Qobj(exc_basis0)  # ground,1h-mid,1h-edge,1l-mid,1l-edge,2h-bound,2h,hl-bound,h0l,l0h,2lbound,2l
        exciton_space = exc_basis * exc_basis.dag()  # ground state of Hamiltonian
        rho = tensor(fock_, exciton_space)

        wc = en_cav / hbar
        wz_h, wz_l = en_exc / hbar  # atom resonant frequencies

        a = tensor([destroy(M)] + [qeye(12)])

        H_cav = hbar * (wc * a.dag() * a)  # cavity term

        # each individual lowering operators
        h_m_low = np.zeros_like(exc_basis0); h_m_low[0, 1] = 1; h_m_low = tensor(qeye(M), Qobj(h_m_low)) # lowers 0h0 to g
        h_e_low = np.zeros_like(exc_basis0); h_e_low[0, 2] = 1; h_e_low = tensor(qeye(M), Qobj(h_e_low)) # lowers 00h to g
        l_m_low = np.zeros_like(exc_basis0); l_m_low[0, 3] = 1; l_m_low = tensor(qeye(M), Qobj(l_m_low)) # lowers 0l0 to g
        l_e_low = np.zeros_like(exc_basis0); l_e_low[0, 4] = 1; l_e_low = tensor(qeye(M), Qobj(l_e_low)) # lowers 00l to g

        hh_m_low = np.zeros_like(exc_basis0); hh_m_low[[1,2],[5,5]] = 1 / np.sqrt(2);
        hh_m_low = tensor(qeye(M), Qobj(hh_m_low)) # lowers 0hh to 0h0 + 00h
        hh_e_low = np.zeros_like(exc_basis0); hh_e_low[2, 6] = 1;
        hh_e_low = tensor(qeye(M), Qobj(hh_e_low)) # lowers h0h to 00h
        ll_m_low = np.zeros_like(exc_basis0); ll_m_low[[3,4], [10,10]] = 1 / np.sqrt(2);
        ll_m_low = tensor(qeye(M), Qobj(ll_m_low)) # lowers 0ll to 0l0 + 00l
        ll_e_low = np.zeros_like(exc_basis0); ll_e_low[4, 11] = 1;
        ll_e_low = tensor(qeye(M), Qobj(ll_e_low)) # lowers l0l to 00l

        hl_m_h_low = np.zeros_like(exc_basis0); hl_m_h_low[4,7] = 1 / np.sqrt(2);
        hl_m_h_low = tensor(qeye(M), Qobj(hl_m_h_low))  # lowers 0hl to 00l
        hl_m_l_low = np.zeros_like(exc_basis0); hl_m_l_low[1,7] = 1 / np.sqrt(2);
        hl_m_l_low = tensor(qeye(M), Qobj(hl_m_l_low))  # lowers 0hl to 0h0
        lh_m_h_low = np.zeros_like(exc_basis0); lh_m_h_low[3,8] = 1 / np.sqrt(2);
        lh_m_h_low = tensor(qeye(M), Qobj(lh_m_h_low))  # lowers 0lh to 0l0
        lh_m_l_low = np.zeros_like(exc_basis0); lh_m_l_low[2,8] = 1 / np.sqrt(2);
        lh_m_l_low = tensor(qeye(M), Qobj(lh_m_l_low))  # lowers 0lh to 00h
        hl_e_h_low = np.zeros_like(exc_basis0); hl_e_h_low[4,9] = 1 / np.sqrt(2);
        hl_e_h_low = tensor(qeye(M), Qobj(hl_e_h_low))  # lowers h0l to 00l
        hl_e_l_low = np.zeros_like(exc_basis0); hl_e_l_low[2,9] = 1 / np.sqrt(2);
        hl_e_l_low = tensor(qeye(M), Qobj(hl_e_l_low))  # lowers h0l to h00

        # group lowering operators
        exc_h_low = h_m_low + h_e_low + hh_m_low + hh_e_low + hl_m_l_low + hl_e_l_low + lh_m_h_low
        exc_l_low = l_m_low + l_e_low + ll_m_low + ll_e_low + hl_m_h_low + hl_e_h_low + lh_m_l_low
        exc_low = exc_h_low + exc_l_low

        H_exc = (hbar * wz_h * (h_m_low.dag() * h_m_low + h_e_low.dag() * h_e_low) +  # h pop
                 hbar * wz_l * (l_m_low.dag() * l_m_low + l_e_low.dag() * l_e_low) +  # l pop
                 hbar * (2 * wz_h - hh_bind / 2) * hh_m_low.dag() * hh_m_low +  # hh-m pop
                 hbar * 2 * wz_h * hh_e_low.dag() * hh_e_low +  # hh-e pop
                 hbar * (2 * wz_l - ll_bind / 2) * ll_m_low.dag() * ll_m_low +  # ll-m pop
                 hbar * 2 * wz_l * ll_e_low.dag() * ll_e_low +  # ll-e pop
                 hbar * (wz_h + wz_l) * (hl_m_h_low.dag() * hl_m_h_low + hl_m_l_low.dag() * hl_m_l_low) +  # hl-m pop
                 hbar * (wz_h + wz_l) * (lh_m_h_low.dag() * lh_m_h_low + lh_m_l_low.dag() * lh_m_l_low) +  # lh-m pop
                 hbar * (wz_h + wz_l - hl_bind / 2) * (hl_e_h_low.dag() * hl_e_h_low + hl_e_l_low.dag() * hl_e_l_low))  # hl-e pop

        # only 1 interacts with the cavity
        if model == "no_rw":
            if dark == "light":
                H_int = hbar * (a + a.dag()) * (exc_h_low + exc_h_low.dag())
            else:
                H_int = hbar * (a + a.dag()) * (exc_l_low + exc_l_low.dag())
        elif model == "rw":
            if dark == "light":
                H_int = hbar * (a.dag() * exc_h_low + a * exc_h_low.dag())
            else:
                H_int = hbar * (a.dag() * exc_l_low + a * exc_l_low.dag())

        H_coupling = h_m_low.dag() * l_m_low + h_e_low.dag() * l_e_low # H-L coupling, no c.c at first; then * (1 + .dag())
        H_coupling += hh_m_low.dag() * (hl_m_l_low + lh_m_l_low) + hh_e_low.dag() * hl_e_l_low
        H_coupling += hl_m_h_low.dag() * ll_m_low + lh_m_h_low.dag() * ll_m_low + hl_e_h_low.dag() * ll_e_low
        H_coupling += H_coupling.dag()
        print("H_coupling", H_coupling)

        H0 = H_cav + H_exc  # diagonal hamiltonian
        H = H0 + g * H_int + J * H_coupling  # total hamiltonian
        print("H0", H0)
        print("H_int", H_int)
        print("H", H)

        mud = muc * (a + a.dag()) + muz * (exc_low + exc_low.dag())
        ad = muc * a + muz * exc_low
        print("mud", mud)
        print("ad", ad)

        # collapse operators: cavity relaxation, cavity exc., collective dephasing, atomic relaxation, atomic exc.
        c_cav_rel = np.sqrt(kappa * (n_th + 1)) * a
        c_cav_exc = np.sqrt(kappa * n_th) * a.dag()
        c_col_dep = np.sqrt(gamma_phase) * H_exc
        c_ato_rel = np.sqrt(gamma_decay * (n_th + 1)) * exc_low
        c_ato_exc = np.sqrt(gamma_decay * n_th) * exc_low.dag()

        c_swap = h_m_low.dag() * h_e_low + h_e_low.dag() * h_m_low + l_m_low.dag() * l_e_low + l_e_low.dag() * l_m_low
        c_swap += hh_m_low.dag() * hh_e_low + hh_e_low.dag() * hh_m_low + ll_m_low.dag() * ll_e_low + ll_e_low.dag() * ll_m_low
        c_swap += hl_e_h_low.dag() * hl_m_h_low + hl_e_l_low.dag() * lh_m_l_low
        c_swap += lh_m_l_low.dag() * hl_e_l_low + hl_m_h_low.dag() * hl_e_h_low
        # print("c_swap", c_swap)

        c_ops = [c_cav_rel, c_cav_exc, c_col_dep, c_swap]

    if N > 3:

        # setting up dm
        fock_ = fock_dm(M, 0)
        exc_basis0 = np.zeros((9, 9));
        exc_basis0[0, 0] = 1;
        exc_basis = Qobj(exc_basis0)  # ground,1h,1l,2h-bound,2h,hl-bound,hl,2l-bound,2l
        exciton_space = exc_basis * exc_basis.dag()
        rho = tensor(fock_, exciton_space)  # ground state of Hamiltonian

        wc = en_cav / hbar
        wz_h, wz_l = en_exc / hbar  # atom resonant frequencies

        a = tensor([destroy(M)] + [qeye(9)])

        H_cav = hbar * (wc * a.dag() * a)  # cavity term

        # each individual lowering operators
        h_low = np.zeros_like(exc_basis0); h_low[0, 1] = 1;
        h_low = tensor(qeye(M), Qobj(h_low))  # lowers h to g
        l_low = np.zeros_like(exc_basis0); l_low[0, 2] = 1;
        l_low = tensor(qeye(M), Qobj(l_low))  # lowers l to g

        hhb_low = np.zeros_like(exc_basis0); hhb_low[[1, 2], [5, 5]] = 1 / np.sqrt(2);
        hhb_low = tensor(qeye(M), Qobj(hhb_low))  # lowers hh-b to h
        hhf_low = np.zeros_like(exc_basis0); hhf_low[2, 6] = 1;
        hhf_low = tensor(qeye(M), Qobj(hhf_low))  # lowers hh-f to h
        llb_low = np.zeros_like(exc_basis0); llb_low[[3, 4], [10, 10]] = 1 / np.sqrt(2);
        llb_low = tensor(qeye(M), Qobj(llb_low))  # lowers ll-b to l
        llf_low = np.zeros_like(exc_basis0); llf_low[4, 11] = 1;
        llf_low = tensor(qeye(M), Qobj(llf_low))  # lowers ll-f to l

        hlb_h_low = np.zeros_like(exc_basis0); hlb_h_low[4, 7] = 1 / np.sqrt(2);
        hlb_h_low = tensor(qeye(M), Qobj(hlb_h_low))  # lowers hl-b to h
        hlf_h_low = np.zeros_like(exc_basis0); hlf_h_low[1, 7] = 1 / np.sqrt(2);
        hlf_h_low = tensor(qeye(M), Qobj(hlf_h_low))  # lowers hl-f to h
        hlb_l_low = np.zeros_like(exc_basis0); hlb_l_low[3, 8] = 1 / np.sqrt(2);
        hlb_l_low = tensor(qeye(M), Qobj(hlb_l_low))  # lowers hl-b to l
        hlf_l_low = np.zeros_like(exc_basis0); hlf_l_low[2, 8] = 1 / np.sqrt(2);
        hlf_l_low = tensor(qeye(M), Qobj(hlf_l_low))  # lowers hl-f to l


        # group lowering operators
        exc_h_low = h_m_low + h_e_low + hh_m_low + hh_e_low + hl_m_l_low + hl_e_l_low + lh_m_h_low
        exc_l_low = l_m_low + l_e_low + ll_m_low + ll_e_low + hl_m_h_low + hl_e_h_low + lh_m_l_low
        exc_low = exc_h_low + exc_l_low

        H_exc = (hbar * wz_h * (h_m_low.dag() * h_m_low + h_e_low.dag() * h_e_low) +  # h pop
                 hbar * wz_l * (l_m_low.dag() * l_m_low + l_e_low.dag() * l_e_low) +  # l pop
                 hbar * (2 * wz_h - hh_bind / 2) * hh_m_low.dag() * hh_m_low +  # hh-m pop
                 hbar * 2 * wz_h * hh_e_low.dag() * hh_e_low +  # hh-e pop
                 hbar * (2 * wz_l - ll_bind / 2) * ll_m_low.dag() * ll_m_low +  # ll-m pop
                 hbar * 2 * wz_l * ll_e_low.dag() * ll_e_low +  # ll-e pop
                 hbar * (wz_h + wz_l) * (hl_m_h_low.dag() * hl_m_h_low + hl_m_l_low.dag() * hl_m_l_low) +  # hl-m pop
                 hbar * (wz_h + wz_l) * (lh_m_h_low.dag() * lh_m_h_low + lh_m_l_low.dag() * lh_m_l_low) +  # lh-m pop
                 hbar * (wz_h + wz_l - hl_bind / 2) * (hl_e_h_low.dag() * hl_e_h_low + hl_e_l_low.dag() * hl_e_l_low))  # hl-e pop

        # only 1 interacts with the cavity
        if model == "no_rw":
            if dark == "light":
                H_int = hbar * (a + a.dag()) * (exc_h_low + exc_h_low.dag())
            else:
                H_int = hbar * (a + a.dag()) * (exc_l_low + exc_l_low.dag())
        elif model == "rw":
            if dark == "light":
                H_int = hbar * (a.dag() * exc_h_low + a * exc_h_low.dag())
            else:
                H_int = hbar * (a.dag() * exc_l_low + a * exc_l_low.dag())

        H_coupling = h_m_low.dag() * l_m_low + h_e_low.dag() * l_e_low # H-L coupling, no c.c at first; then * (1 + .dag())
        H_coupling += hh_m_low.dag() * (hl_m_l_low + lh_m_l_low) + hh_e_low.dag() * hl_e_l_low
        H_coupling += hl_m_h_low.dag() * ll_m_low + lh_m_h_low.dag() * ll_m_low + hl_e_h_low.dag() * ll_e_low
        H_coupling += H_coupling.dag()
        # print("H_coupling", H_coupling)

        H0 = H_cav + H_exc  # diagonal hamiltonian
        H = H0 + g * H_int + J * H_coupling  # total hamiltonian
        # print("H0", H0)
        # print("H_int", H_int)
        # print("H", H)

        mud = muc * (a + a.dag()) + muz * (exc_low + exc_low.dag())
        ad = muc * a + muz * exc_low
        # print("mud", mud)
        # print("ad", ad)

        # collapse operators: cavity relaxation, cavity exc., collective dephasing, atomic relaxation, atomic exc.
        c_cav_rel = np.sqrt(kappa * (n_th + 1)) * a
        c_cav_exc = np.sqrt(kappa * n_th) * a.dag()
        c_col_dep = np.sqrt(gamma_phase) * H_exc
        c_ato_rel = np.sqrt(gamma_decay * (n_th + 1)) * exc_low
        c_ato_exc = np.sqrt(gamma_decay * n_th) * exc_low.dag()

        c_swap = h_m_low.dag() * h_e_low + h_e_low.dag() * h_m_low + l_m_low.dag() * l_e_low + l_e_low.dag() * l_m_low
        c_swap += hh_m_low.dag() * hh_e_low + hh_e_low.dag() * hh_m_low + ll_m_low.dag() * ll_e_low + ll_e_low.dag() * ll_m_low
        c_swap += hl_e_h_low.dag() * hl_m_h_low + hl_e_l_low.dag() * lh_m_l_low
        c_swap += lh_m_l_low.dag() * hl_e_l_low + hl_m_h_low.dag() * hl_e_h_low
        # print("c_swap", c_swap)

        c_ops = [c_cav_rel, c_cav_exc, c_col_dep, c_swap]

    title = "exc_exc_model_energies"

    print("dimensionality of Hilbert-space: ", H.shape)

    # setting up system
    sys = System(H=H, rho=rho, a=ad, u=mud, c_ops=c_ops, diagonalize=True)

    en, T = H.eigenstates()
    print("en", en)

    print("system has been intialized")

    # # calculate the expectation value of the number of photons in the cavity
    # n_vec = expect(a.dag() * a, rho)
    # # print("expectation of number of photons: ", n_vec)
    # n2_vec = expect(a.dag() * a*a.dag() * a, rho)
    # a_vec = expect(a, rho)

    title = "exc_exc_model_energies"

    print("dimensionality of Hilbert-space: ", H.shape)

    # setting up system
    sys = System(H=H, rho=rho, a=ad, u=mud, c_ops=c_ops, diagonalize=True)

    en, T = H.eigenstates()

    print("system has been intialized")

    # # calculate the expectation value of the number of photons in the cavity
    # n_vec = expect(a.dag() * a, rho)
    # # print("expectation of number of photons: ", n_vec)
    # n2_vec = expect(a.dag() * a*a.dag() * a, rho)
    # a_vec = expect(a, rho)

    if order == 1:
        d0 = 100

        # generating spectra
        dipole, t_list, spec, freq = sys.linear_spec(d0, r=10, title_graph=title, plot_graph=False, dir=directory)

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

        # sys.diagram_donkey([0, 100, 100+time2, 200+time2], [R3], r=10, title_graph=None, plot_graph=False, dir=directory)

        print("finished setup stage")

        dipole, t_list, spec, freq = sys.linear_spec(10000, r=1)
        plt.figure(figsize=(12, 6))
        plt.plot(t_list, np.real(dipole), label="$J_{0} = 0 \ eV$")  # real part of the dipole
        plt.xlim(0, 80)
        plt.xlabel("Time (fs)")
        plt.ylabel("Dipole")
        plt.legend(fontsize=14)
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(freq, np.real(spec), label="$J_{0} = 0 \ eV$")  # real part of the spectrum
        plt.xlim(0.5, 3.0)
        plt.xticks(np.arange(0.5, 3.0, 0.1))
        for i in H.eigenenergies():
            plt.axvline(x=i, color='red', linestyle='--')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Spectrum")
        plt.legend(fontsize=14)
        plt.grid()
        plt.show()

        # generating 2Dcoherence response for rephasing diagrams
        time_delays = [100, 20, 100]
        scan_id = [0, 2]
        response_list = []
        diagrams = rephasing + nonrephasing
        for k in range(6):
            states, t1, t2, dipole = sys.coherence2d(time_delays, diagrams[k], scan_id, r=1, parallel=True)
            print('diagram ', k, ' done')
            response_list.append(1j * dipole)
        spectra_list, extent, f1, f2 = sys.spectra(np.imag(response_list), resolution=1)

        rephasing_spectra = spectra_list[:3]
        rephasing_spectra.append(np.sum(spectra_list[:3], 0))
        pf.silva_plot_contourf(rephasing_spectra, f1, f2, labels=['E emission', 'E absorption'],
                  scale='linear', color_map='PuOr', title_list = ['$R_1$', '$R_2$', '$R_3$', '$R_{rephasing}$'],
                  center_scale=False, plot_sum=False, plot_quadrant='2', zoom_coor=[-2.8,-2,2,2.8],
                  invert_y=False, diagonals=[True, False], nlevels=15,
                  title_graph="N3 M3 bind 0982", plot_graph=True, direc=directory)

spectrum_var(N=2, M=2, gamma_hl=0.1, l_str=0.01)
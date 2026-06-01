# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:32:xx 2026

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

def spectrum_var(order=3, E0=1.0, E=1.1, g=0.05, muc=1.0, muz=1.0,
                 kappa=0.05, gamma_phase=0.15, gamma_decay=0.15, M=2, N=1,
                 model="no_rw", n_th=0.25, decay="no_atomic", T2=None, directory="results",
                 anim="no"):    #choose order
    """
        Plot multiple spectra with real, imaginary and abs values
        :param order: linear(1) or 3rd(3). Must be int data type
        :param E0:
        :param E:
        :param g: # initial coupling strength for cavity mode-spin interaction
        :param muc: dipole strength for the cavity
        :param muz: dipole strength for the particle
        :param kappa: cavity decay strength
        :param gamma_phase: atomic dephasing strength
        :param gamma_decay: atomic decay strength
        :param M: # number of fock basis states for cavity mode. Use larger value for stronger couplings
        :param N: # number of spins
        :param model: "no_rw" for no rotating wave approximation, "rw" for rotating wave approximation
        :param n_th: average number thermal photons in the bath coupling to the resonator
        :param decay: "no_atomic" "atomic"
        :param T2: 2nd order time delay
        :param directory: result directory
        :param anim: animate
        :return: Does not return anything unless anim

        """

    wc = E0/hbar # cavity resonant frequency
    wz = E/hbar # atom resonant frequency
    gc = np.sqrt(wz * wc)/2  # critical coupling strength
    # print("critical coupling strength = ", gc)

    s = N/2  # Total J for spins.
    n = N+1  # dimensionality of total spin operator.

    a = tensor(destroy(M), qeye(n))
    Sp = tensor(qeye(M), -jmat(s, '+'))
    Sm = tensor(qeye(M), -jmat(s, '-'))
    Sx = tensor(qeye(M), -jmat(s, 'x'))
    Sy = tensor(qeye(M), -jmat(s, 'y'))
    Sz = tensor(qeye(M), -jmat(s, 'z'))
    mud = muc * (a + a.dag()) + muz * Sx
    ad = a + Sm


    H0 = hbar * (wc * a.dag() * a + wz * Sz)    # default basic Hamiltonian
    if model == "no_rw": # no rotating wave approx
        H1 = hbar * (a + a.dag()) * Sx/np.sqrt(N)  # intra-cavity interaction term
        tit = 1
    elif model == "rw": # with rotating wave approx
        H1 = hbar * (a * Sp + a.dag() * Sm) / np.sqrt(N)  # intra-cavity interaction term
        tit = 9

    H = H0 + g * H1 # total hamiltonian
    # print(hbar * wc * a.dag() * a)
    # print(hbar * wz * Sz)
    print(H0)
    print(H)

    # collapse operators: cavity relaxation, cavity exc., collective dephasing, atomic relaxation, atomic exc.
    c_cav_rel = np.sqrt(kappa * (n_th + 1)) * a
    c_cav_exc = np.sqrt(kappa * n_th) * a.dag()
    c_col_dep = np.sqrt(gamma_phase) * Sz
    c_ato_rel = np.sqrt(gamma_decay * (n_th + 1)) * Sm
    c_ato_exc =np.sqrt(gamma_decay * n_th) * Sp

    if decay == "no_atomic":
        c_ops = [c_cav_rel, c_cav_exc, c_col_dep]
    elif decay == "atomic":
        c_ops = [c_cav_rel, c_cav_exc, c_col_dep, c_ato_rel, c_ato_exc]
        tit += 1
    if T2 == None:
        T2 = 10
    else:
        tit += 2
    if g == None:
        g = 0.05
    else:
        tit += 4

    titles = {1: "no rw, base collapse",
              2: "no rw, atomic collapse",
              3: "no rw, base collapse, T2="+str(T2),
              4: "no rw, atomic collapse, T2="+str(T2),
              5: "no rw, base collapse, g="+str(g),
              6: "no rw, atomic collapse, g="+str(g),
              7: "no rw, base collapse, T2=" + str(T2) +", g="+str(g),
              8: "no rw, atomic collapse, T2=" + str(T2) +", g="+str(g),
              9: "rw, base collapse",
              10: "rw, atomic collapse",
              11: "no rw, base collapse, T2="+str(T2),
              12: "no rw, atomic collapse, T2="+str(T2),
              13: "rw, base collapse, g="+str(g),
              14: "rw, atomic collapse, g="+str(g),
              15: "no rw, base collapse, T2=" + str(T2) +", g="+str(g),
              16: "no rw, atomic collapse, T2=" + str(T2) +", g="+str(g),
              }

    title = "M=" + str(M) + ", N=" + str(N) + ", " + titles[tit]
    # titles = {"y_rw": {"base": 1,
    #                    "atomic": 2
    #                    },
    #           "n_rw": {"base": 1,
    #                    "atomic": 2
    #                    }
    #           }


    # print("dimensionality of Hilbert-space: ", H.shape)

    # setting up system
    rho = tensor(fock_dm(M, 0), fock_dm(N+1, 0))  # ground state of Hamiltonian
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
        if anim != "no":
            return pf.silva_plot_contourf(rephasing_spectra, f1, f2, labels=['E emission', 'E absorption'],
                      scale='linear', color_map='PuOr',
                      center_scale=False, plot_sum=False, plot_quadrant='2',
                      zoom_coor=[-2.8,-2,2,2.8], invert_y=False, diagonals=[True, False], nlevels=10,
                      title_graph=None, plot_graph=False, dir=directory, anim=anim)
        else:
            pf.silva_plot_contourf(rephasing_spectra, f1, f2, labels=['E emission', 'E absorption'],
                      scale='linear', color_map='PuOr', title_list = ['$R_1$', '$R_2$', '$R_3$', '$R_{rephasing}$'],
                      center_scale=False, plot_sum=False, plot_quadrant='2', zoom_coor=[-2.8,-2,2,2.8],
                      invert_y=False, diagonals=[True, False], nlevels=10,
                      title_graph=None, plot_graph=True, dir=directory)


# animation -----------------------------------------------------------------------------------------------------------------

# # R1 + R2 + R3 abs animation
fig, ax = plt.subplots(1, 2, num="anim", gridspec_kw={"width_ratios": [1, 0.06]})
# ax = fig.add_subplot(111)
ax[0].set_xlabel("E emission")
ax[0].set_ylabel("E absorption")
artists = []
num = 10
ls = np.linspace(10, 200, num)
for ii in range(len(ls)):
    print(ii)
    print(ls[ii])
    data, scan_range, diag_range, norm, v_range = spectrum_var(order=3, E0=1., E=1.1, g=0.05, muc=1.0, muz=1.0,
                          kappa=0.05, gamma_phase=0.1, gamma_decay=0.1, M=2, N=1,
                          model="rw", n_th=0.25, decay="no_atomic", T2=ls[ii], directory="comparaisons", anim="yes")
    plt.figure("anim")
    if ii > 0:
        # cbar.remove()
        for handle in diags:
            handle.remove()
    txt = plt.text(0.5, 1.01, "T2 = {0}".format(str(round(ls[ii], 3))), ha="center", va="bottom", color="black",
                       transform=ax[0].transAxes, fontsize="large")

    sm = cm.ScalarMappable(norm=norm, cmap='PuOr')
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax[1])
    ticks = np.linspace(*v_range, num=5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([])
    tick = []
    for i in range(len(ticks)):
        tick.append(plt.text(2.2, 0.98-i*.25, str(round(ticks[-i-1],1)), ha="center", va="bottom", color="black",
                   transform=ax[1].transAxes, fontsize="medium"))

    diags = ax[0].plot([diag_range[-1][0], diag_range[-1][1]], [diag_range[-1][3], diag_range[-1][2]], '--',
                     color="black", linewidth=0.5)

    im1 = ax[0].contourf(*data, levels=20, cmap="PuOr", norm=norm)
    im2 = ax[0].contour(*data, levels=20, colors='k', linewidths=0.4, alpha=0.7)

    artists.append([im1, im2, txt, *tick])

cbar.ax.set_yticklabels([])
ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=4000/num)

writergif = animation.PillowWriter(fps=num/4, bitrate=2000)
ani.save("comparaisons/animations/Jaynes_Cummings_T2.gif", writer=writergif)
plt.show()
#
# spectrum_var(order=3, E0=1.0, E=1.1, g=0., muc=1.0, muz=1.0,
#                  kappa=0.05, gamma_phase=0.1, gamma_decay=0.1, M=2, N=1,
#                  model="no_rw", n_th=0.25, decay="no_atomic", directory="T2_range_3rd")

# fin animation -----------------------------------------------------------------------------------------------------------------
# # interactive 3D -----------------------------------------------------------------------------------------------------------------
#
# data, scan_range, diag_range, norm, v_range = spectrum_var(order=3, E0=1., E=1.1, g=0.05, muc=1.0, muz=1.0,
#                           kappa=0.1, gamma_phase=0.1, gamma_decay=0.1, M=2, N=1,
#                           model="no_rw", n_th=0.25, decay="no_atomic", directory="comparaisons", anim="yes")
#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X, Y, Z = data
# X = [np.float64(i) for i in X]
# X, Y = np.meshgrid(X, Y) # NECESSARY IF NOT "All" FOR QUADRANTS
#
# # ax.contour(*data, cmap=cm.coolwarm)  # Plot contour curves
# # ax.plot_surface(X, Y, Z, edgecolor='orange', lw=0.5, rstride=8, cstride=8, alpha=0.3)
# sm = ax.contour(X, Y, Z, cmap='PuOr', levels=40)
# cbar = fig.colorbar(sm, shrink=0.5, aspect=5)
# ax.set_xlabel("E absorption")
# ax.set_ylabel("E emission")
# ax.set_zlabel("absolute value")
# plt.show()
#
# # fin interactive 3D -----------------------------------------------------------------------------------------------------------------



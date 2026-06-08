import numpy as np
import qutip as qt
import scipy.linalg as la
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def get_H_and_Js(k, t1, t2):
    """Retourne l'Hamiltonien et les opérateurs de courant pour k et -k."""
    H_mat = np.array([[0, t1 + t2 * np.exp(-1j * k)],
                      [t1 + t2 * np.exp(1j * k), 0]])
    
    # J(k) = dH/dk
    J_k_mat = np.array([[0, -1j * t2 * np.exp(-1j * k)],
                        [1j * t2 * np.exp(1j * k), 0]])
    
    # J(-k)
    J_minusk_mat = np.array([[0, -1j * t2 * np.exp(1j * k)],
                             [1j * t2 * np.exp(-1j * k), 0]])
    
    return qt.Qobj(H_mat), qt.Qobj(J_k_mat), qt.Qobj(J_minusk_mat)

def get_thermal_state(H, T, mu=0.0):
    """Calcule la matrice densité d'équilibre de Fermi-Dirac."""
    evals, evecs = H.eigenstates()
    
    # Distribution de Fermi-Dirac
    f_occ = 1.0 / (np.exp((evals - mu) / T) + 1.0)
    
    # Reconstruction de la matrice densité
    rho_eq = sum([f_occ[i] * evecs[i] * evecs[i].dag() for i in range(len(evals))])
    return rho_eq

def sigma_frohlich(omega, omega_LO, lam, eta, T):
    """Calcule l'auto-énergie (scalaire complexe)."""
    n_ph = 1.0 / (np.exp(omega_LO / T) - 1.0)
    term1 = (n_ph + 1.0) / ((omega - omega_LO) + 1j * eta)
    term2 = n_ph / ((omega + omega_LO) + 1j * eta)
    return (lam**2) * (term1 + term2)

def get_L_eff(H, omega, omega_LO, lam, eta, T):
    """Construit le superopérateur de Liouville effectif pour une fréquence donnée."""
    # Hamiltonien effectif non-hermitien
    Sigma = sigma_frohlich(omega, omega_LO, lam, eta, T)
    H_eff = H + Sigma * qt.qeye(2)
    
    # L = H_eff (x) I - I (x) H_eff^T
    # Dans QuTiP, spre agit à gauche, spost agit à droite (inclut la transposition de Liouville)
    L_super = qt.spre(H_eff) - qt.spost(H_eff)
    return L_super

def get_green_functions(L_w1, L_w3, tau2, w1, w3, eta):
    """Calcule les trois propagateurs G1(w1), G2(tau2), G3(w3)."""
    # Matrice identité 4x4
    I_4 = np.eye(4)
    
    # Extraction des matrices denses 4x4 (espace de Liouville) depuis QuTiP
    L_w1_mat = L_w1.full()
    L_w3_mat = L_w3.full()
    
    # Inversion pour le domaine fréquentiel
    G1_mat = la.inv((w1 + 1j * eta) * I_4 - L_w1_mat)
    G3_mat = la.inv((w3 + 1j * eta) * I_4 - L_w3_mat)
    
    # Exponentielle de matrice pour le temps de population tau2
    G2_mat = la.expm(-1j * L_w1_mat * tau2) 
    # Attention: ton code utilise L[w1] pour tau2, je garde cette convention.
    
    return G1_mat, G2_mat, G3_mat

def calc_rephasing_S3_k(k, w3, w1, tau2, params):
    t1, t2 = params["t1"], params["t2"]
    eta, T = params["Eta"], params["T"]
    wLO, lam, mu = params["omegaLO"], params["lambda"], params["mu"]
    
    H, J_k, J_minusk = get_H_and_Js(k, t1, t2)
    rho_eq = get_thermal_state(H, T, mu)
    
    # --- Construction des superopérateurs (selon Mathematica) ---
    # Interaction 1 (k)
    JL1 = qt.spre(J_k).full()
    JR1 = qt.spost(J_k).full()
    
    # Interaction 2 (-k)
    JL2 = qt.spre(J_minusk).full()
    JR2 = qt.spost(J_minusk).full()
    
    # Interaction 3 (-k)
    JR3 = qt.spost(J_minusk).full()
    
    # Opérateur de trace finale (k)
    JL_out = qt.spre(J_k).full()
    
    # --- Propagateurs ---
    L_w1 = get_L_eff(H, w1, wLO, lam, eta, T)
    L_w3 = get_L_eff(H, w3, wLO, lam, eta, T)
    G1, G2, G3 = get_green_functions(L_w1, L_w3, tau2, w1, w3, eta)
    
    rho_vec = rho_eq.full().flatten('F')
    
    # --- Voies Liouvilliennes ---
    path_GSB = G3 @ JR3 @ G2 @ JL2 @ G1 @ JL1 @ rho_vec
    path_SE  = G3 @ JR3 @ G2 @ JL2 @ G1 @ JR1 @ rho_vec
    path_ESA = G3 @ JR3 @ G2 @ JR2 @ G1 @ JR1 @ rho_vec
    
    I_vec = np.eye(2).flatten('F')
    
    # On prend la trace finale avec JL_out
    tr_GSB = -1j * np.dot(I_vec, JL_out @ path_GSB)
    tr_SE  = -1j * np.dot(I_vec, JL_out @ path_SE)
    tr_ESA = -1j * np.dot(I_vec, JL_out @ path_ESA)
    
    total_trace = tr_GSB + tr_SE + tr_ESA
    
    # (A conservé pour la parité avec l'output Mathematica)
    A = -(1/np.pi) * np.imag(np.trace(G3))
    
    return total_trace, A

def integrate_k_response(w1, w3, tau2, params, k_points=100, n_jobs=-1):
    """
    Intègre la réponse S3 sur la zone de Brillouin complète.
    n_jobs=-1 utilise tous les coeurs de calcul disponibles.
    """
    k_array = np.linspace(-np.pi, np.pi, k_points)
    dk = k_array[1] - k_array[0]
    
    # Calcul parallèle sur tous les points k
    results = Parallel(n_jobs=n_jobs, require='sharedmem')(
    delayed(calc_rephasing_S3_k)(k, w3, w1, tau2, params) for k in k_array
)
    
    # Séparation des résultats (total_trace, A) et intégration (somme * dk)
    S3_k_list, A_k_list = zip(*results)
    
    S3_macro = np.sum(S3_k_list) * dk / (2 * np.pi)
    A_macro = np.sum(A_k_list) * dk / (2 * np.pi)
    
    return S3_macro, A_macro

def generate_2D_spectrum(w1_list, w3_list, tau2, params, k_points=50):
    """Génère la grille 2D du signal de rephasage."""
    S3_grid = np.zeros((len(w3_list), len(w1_list)), dtype=complex)
    
    for i, w3 in enumerate(w3_list):
        for j, w1 in enumerate(w1_list):
            S3_macro, _ = integrate_k_response(w1, w3, tau2, params, k_points)
            S3_grid[i, j] = S3_macro
            
    return S3_grid

if __name__ == '__main__':
    # --- Paramètres alignés sur Mathematica ---
    # Grille restreinte [0.5, 1.5] pour éviter le pic de population à w=0
    w_list = np.linspace(0.5, 1.5, 50) 
    tau2_test = 3.0  # Mis à jour selon ton fichier
    
    params_topo = {"t1": 1.0, "t2": 1.5, "Eta": 0.05, "T": 0.01, "mu": 0.0, "omegaLO": 1.0, "lambda": 0.1}
    params_triv = {"t1": 1.0, "t2": 0.5, "Eta": 0.05, "T": 0.01, "mu": 0.0, "omegaLO": 1.0, "lambda": 0.1}

    print("Calcul de la phase topologique (peut prendre du temps)...")
    S3_topo = generate_2D_spectrum(w_list, w_list, tau2_test, params_topo, k_points=50)

    print("Calcul de la phase triviale...")
    S3_triv = generate_2D_spectrum(w_list, w_list, tau2_test, params_triv, k_points=50)

    # --- Section Matplotlib ---
  
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Définir une échelle d'amplitude symétrique stricte basée sur le signal Topo
    limite_amplitude = np.max(np.abs(np.imag(S3_topo)))
    
    # 2. Plot Phase Topologique 2D
    ax = axes[0]
    # On utilise 'RdBu_r' (Rouge/Bleu) ou 'seismic' qui est idéal pour les spectres 2D
    # vmin et vmax forcent le blanc (ou le milieu de la colormap) à être exactement à 0
    c1 = ax.contourf(w_list, w_list, np.imag(S3_topo), 60, 
                     cmap='RdBu_r', vmin=-limite_amplitude, vmax=limite_amplitude) 
    ax.set_title(r"Imag/Topo ($t_2 > t_1$)")
    ax.set_xlabel(r"$\omega_1$")
    ax.set_ylabel(r"$\omega_3$")
    fig.colorbar(c1, ax=ax)

    # 3. Plot Phase Triviale 2D
    ax = axes[1]
    # On applique EXACTEMENT la même échelle (vmin, vmax) au graphe trivial
    c2 = ax.contourf(w_list, w_list, np.imag(S3_triv), 60, 
                     cmap='RdBu_r', vmin=-limite_amplitude, vmax=limite_amplitude)
    ax.set_title(r"Imag/Triv ($t_1 > t_2$)")
    ax.set_xlabel(r"$\omega_1$")
    ax.set_ylabel(r"$\omega_3$")
    fig.colorbar(c2, ax=ax)

    plt.tight_layout()
    plt.show()
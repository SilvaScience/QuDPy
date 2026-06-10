import numpy as np
import matplotlib.pyplot as plt

class LiouvilleSpectroscopySolver:
    def __init__(self, params):
        """
        Solveur universel d'espace de Liouville pour la spectroscopie 2D.
        params : Dictionnaire contenant 'Eta' (élargissement) et 'T' (température).
        """
        self.params = params
        self.eta = params.get("Eta", 0.05)
        self.T = params.get("T", 0.01)
        
        # Variables d'état du modèle (remplies par feed_model)
        self.H_eigen = None
        self.J_plus = None
        self.J_minus = None
        self.c_ops = []
        self.dim = None
        self.N_k = None

    def feed_model(self, H_model, J_model):
        """
        Reçoit l'Hamiltonien et le Courant dans la base des sites (N_k, d, d).
        Diagonalise le système et sépare les opérateurs selon la RWA de façon universelle.
        """
        self.N_k, self.dim, _ = H_model.shape
        d = self.dim
        
        # 1. Diagonalisation de l'Hamiltonien sur tous les k-points
        evals, evecs = np.linalg.eigh(H_model)
        
        # Construction propre de H_eigen (diagonale)
        self.H_eigen = np.zeros_like(H_model)
        for i in range(d):
            self.H_eigen[:, i, i] = evals[:, i]
            
        # 2. Rotation de l'opérateur de courant dans la base des états propres
        V_dag = np.conj(evecs.transpose(0, 2, 1))
        J_eigen = V_dag @ J_model @ evecs
        
        # 3. Application universelle de la RWA (indépendante du nombre de bandes)
        self.J_plus = np.zeros_like(J_eigen)
        self.J_minus = np.zeros_like(J_eigen)
        
        # j > i implique une transition vers le haut en énergie (absorption)
        for i in range(d):
            for j in range(d):
                if j > i:
                    self.J_plus[:, j, i] = J_eigen[:, j, i]
                elif j < i:
                    self.J_minus[:, j, i] = J_eigen[:, j, i]

    def set_dissipation(self, c_ops_list):
        """
        Définit les opérateurs de saut de Lindblad.
        c_ops_list : Liste de tuples [(Matrice_N_k_d_d, taux_gamma), ...]
        """
        self.c_ops = c_ops_list

    # ========================================================================
    # ALGÈBRE DE LIOUVILLE VECTORISÉE (BATCHED OVER K)
    # ========================================================================
    def _spre(self, A):
        """Superopérateur agissant à gauche : I_d \otimes A (Ordre Fortran)"""
        I = np.eye(self.dim)
        res = np.einsum('ij,nkl->nikjl', I, A)
        return res.reshape(self.N_k, self.dim**2, self.dim**2)
    
    def _spost(self, A):
        """Superopérateur agissant à droite : A^T \otimes I_d (Ordre Fortran)"""
        I = np.eye(self.dim)
        res = np.einsum('nji,kl->nikjl', A, I)
        return res.reshape(self.N_k, self.dim**2, self.dim**2)

    def _get_lindblad(self, C, gamma):
        """Calcule la matrice de dissipation de Lindblad complète (N_k, d^2, d^2)"""
        C_dag = np.conj(C.transpose(0, 2, 1))
        C_dag_C = C_dag @ C
        return gamma * (self._spre(C) @ self._spost(C_dag) - 
                        0.5 * self._spre(C_dag_C) - 
                        0.5 * self._spost(C_dag_C))

    def _get_L_eff(self, w_probe):
        """Génère le Liouvillien effectif total pour une fréquence donnée"""
        # Partie cohérente : -i [H, \rho]
        L = (self._spre(self.H_eigen) - self._spost(self.H_eigen))
        # Ajout de la relaxation de Lindblad
        for C, gamma in self.c_ops:
            L += 1j * self._get_lindblad(C, gamma)
        return L

    def _get_thermal_state(self):
        """Calcule l'état d'équilibre thermique initial vectorisé (N_k, d^2, 1)"""
        d = self.dim
        mu = self.params.get("mu", 0.0)
        
        if self.T > 0:
            beta = 1.0 / self.T
            energies = np.array([self.H_eigen[:, i, i] for i in range(d)]).T
            exp_factors = np.exp(-beta * (energies - mu))
            rho_diag = exp_factors / np.sum(exp_factors, axis=1, keepdims=True)
        else:
            rho_diag = np.zeros((self.N_k, d))
            rho_diag[:, 0] = 1.0 # Tout dans le fondamental à T=0
            
        rho_vec = np.zeros((self.N_k, d**2, 1), dtype=complex)
        for i in range(d):
            rho_vec[:, i * d + i, 0] = rho_diag[:, i]
        return rho_vec

    # ========================================================================
    # CALCUL DES SPECTRES ET CHEMINS DE FEYNMAN
    # ========================================================================
    def calc_rephasing(self, w3, w1, tau2):
        """Calcul des diagrammes de rephasage (-k1, +k2, +k3)"""
        d2 = self.dim**2
        I_super = np.eye(d2)
        rho_vec = self._get_thermal_state()
        
        # Initialisation des superopérateurs d'interaction
        JL_plus = self._spre(self.J_plus)
        JR_plus = self._spost(self.J_plus)
        JR_minus = self._spost(self.J_minus)
        JL_out = self._spre(self.J_plus + self.J_minus)
        
        # Résolvantes (G1 à -w1 pour le rephasage)
        L_w1 = self._get_L_eff(w1)
        L_w3 = self._get_L_eff(w3)
        L_0  = self._get_L_eff(0.0)
        
        G1 = np.linalg.inv((-w1 + 1j * self.eta) * I_super - L_w1)
        G3 = np.linalg.inv((w3 + 1j * self.eta) * I_super - L_w3)
        
        # Exponentielle matricielle vectorisée via diagonalisation pour tau2
        evals, evecs = np.linalg.eig(-1j * L_0 * tau2)
        G2 = (evecs * np.exp(evals)[:, np.newaxis, :]) @ np.linalg.inv(evecs)
        
        # Évaluation des chemins de Feynman
        path_GSB = G3 @ (JL_plus @ (G2 @ (JR_plus @ (G1 @ (JR_minus @ rho_vec)))))
        path_SE  = G3 @ (JR_plus @ (G2 @ (JL_plus @ (G1 @ (JR_minus @ rho_vec)))))
        path_ESA = G3 @ (JL_plus @ (G2 @ (JL_plus @ (G1 @ (JR_minus @ rho_vec)))))
        
        # Opérateur de trace
        I_vec = np.zeros((self.N_k, 1, d2), dtype=complex)
        for i in range(self.dim):
            I_vec[:, 0, i * self.dim + i] = 1.0
            
        tr_GSB = (I_vec @ (JL_out @ path_GSB)).reshape(-1)
        tr_SE  = (I_vec @ (JL_out @ path_SE)).reshape(-1)
        tr_ESA = (I_vec @ (JL_out @ path_ESA)).reshape(-1)
        
        return -1j * (tr_GSB + tr_SE - tr_ESA)

    def calc_unrephasing(self, w3, w1, tau2):
        """Calcul des diagrammes de non-rephasage (+k1, -k2, +k3)"""
        d2 = self.dim**2
        I_super = np.eye(d2)
        rho_vec = self._get_thermal_state()
        
        JL_plus = self._spre(self.J_plus)
        JR_plus = self._spost(self.J_plus)
        JL_minus = self._spre(self.J_minus)
        JR_minus = self._spost(self.J_minus)
        JL_out = self._spre(self.J_plus + self.J_minus)
        
        L_w1 = self._get_L_eff(w1)
        L_w3 = self._get_L_eff(w3)
        L_0  = self._get_L_eff(0.0)
        
        # G1 est à +w1 pour le non-rephasage
        G1 = np.linalg.inv((w1 + 1j * self.eta) * I_super - L_w1)
        G3 = np.linalg.inv((w3 + 1j * self.eta) * I_super - L_w3)

        
        
        evals, evecs = np.linalg.eig(-1j * L_0 * tau2)
        G2 = (evecs * np.exp(evals)[:, np.newaxis, :]) @ np.linalg.inv(evecs)
        
        
        # Séquence de non-rephasage
        path_GSB = G3 @ (JL_plus @ (G2 @ (JL_minus @ (G1 @ (JL_plus @ rho_vec)))))
        path_SE  = G3 @ (JR_plus @ (G2 @ (JR_minus @ (G1 @ (JL_plus @ rho_vec)))))
        path_ESA = G3 @ (JL_plus @ (G2 @ (JR_minus @ (G1 @ (JL_plus @ rho_vec)))))
        
        I_vec = np.zeros((self.N_k, 1, d2), dtype=complex)
        for i in range(self.dim):
            I_vec[:, 0, i * self.dim + i] = 1.0
            
        tr_GSB = (I_vec @ (JL_out @ path_GSB)).reshape(-1)
        tr_SE  = (I_vec @ (JL_out @ path_SE)).reshape(-1)
        tr_ESA = (I_vec @ (JL_out @ path_ESA)).reshape(-1)
        
        return -1j * (tr_GSB + tr_SE - tr_ESA)

    def generate_2D_spectra(self, w_list, tau2, k_array):
        """
        Scanne la grille de fréquences w1 et w3.
        Effectue la somme sur la zone de Brillouin (intégration sur k).
        Returns: Un dictionnaire contenant les matrices 2D complexes de réponse.
        """
        n_w = len(w_list)
        dk = (k_array[-1] - k_array[0]) / len(k_array)
        
        S3_reph = np.zeros((n_w, n_w), dtype=complex)
        S3_unreph = np.zeros((n_w, n_w), dtype=complex)
        
        for i, w1 in enumerate(w_list):
            for j, w3 in enumerate(w_list):
                # Calcul simultané pour tous les k-points
                vec_reph = self.calc_rephasing(w3, w1, tau2)
                vec_unreph = self.calc_unrephasing(w3, w1, tau2)
                
                # Intégration numérique trapézoïdale / somme sur la zone de Brillouin
                S3_reph[j, i] = np.sum(vec_reph) * dk / (2 * np.pi)
                S3_unreph[j, i] = np.sum(vec_unreph) * dk / (2 * np.pi)
                
        return {
            "rephasing": S3_reph,
            "unrephasing": S3_unreph,
            "absorptive": S3_reph + S3_unreph
        }


class SpectroscopyPlotter:
    def __init__(self, w_list): 
        self.w_list = w_list

    def plot_spectrum(self, S3_topo, S3_triv, levels=60, figsize=(12, 14), save_path=None):
        """
        Affiche une figure 3x2 avec les spectres 2D (Réel, Imag, Abs)
        pour les phases topologique et triviale.
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        w = self.w_list

        # ==========================================
        # ROW 1: REAL PART
        # ==========================================
        self._plot_subplot(axes[0, 0], w, np.real(S3_topo), levels, 'RdBu_r', 
                           r"Real / Topo ($t_2 > t_1$)", ylabel=r"$\omega_3$")
        self._plot_subplot(axes[0, 1], w, np.real(S3_triv), levels, 'RdBu_r', 
                           r"Real / Triv ($t_1 > t_2$)")

        # ==========================================
        # ROW 2: IMAGINARY PART
        # ==========================================
        self._plot_subplot(axes[1, 0], w, np.imag(S3_topo), levels, 'RdBu_r', 
                           r"Imag / Topo ($t_2 > t_1$)", ylabel=r"$\omega_3$")
        self._plot_subplot(axes[1, 1], w, np.imag(S3_triv), levels, 'RdBu_r', 
                           r"Imag / Triv ($t_1 > t_2$)")

        # ==========================================
        # ROW 3: ABSOLUTE VALUE
        # ==========================================
        self._plot_subplot(axes[2, 0], w, np.abs(S3_topo), levels, 'magma', 
                           r"Abs / Topo ($t_2 > t_1$)", xlabel=r"$\omega_1$", ylabel=r"$\omega_3$", vmin=0)
        self._plot_subplot(axes[2, 1], w, np.abs(S3_triv), levels, 'magma', 
                           r"Abs / Triv ($t_1 > t_2$)", xlabel=r"$\omega_1$", vmin=0)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved at: {save_path}")
            
        plt.show()

    def _plot_subplot(self, ax, w, data, levels, cmap, title, xlabel=None, ylabel=None, vmin=None):
        """Méthode privée pour éviter la répétition du code de contourf"""
        lim = np.max(np.abs(data))
        if vmin is None:
            vmin, vmax = -lim, lim
        else:
            vmax = lim
            
        c = ax.contourf(w, w, data, levels, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        plt.colorbar(c, ax=ax)








'''
import numpy as np
from scipy.linalg import expm


class LiouvilleSpectroscopySolver:
    def __init__(self,params):
        self.params = params

        #Model feed to the class
        self.H_eigen = None
        self.J_plus = None
        self.J_minus = None
        self.c_ops = []

    def feed_model(self, H_model, J_model):
        """
        Injection of the matrix, diagonalise and apply the RWA
        """    

        N = H_model.shape[0]
        # Diagonalisation (low energy)

        evals, evecs = np.linalg.eigh(H_model)
        self.H_eigen = np.zeros_like(H_model)
        self.H_eigen [:,0,0] = evals[:,0]
        self.H_eigen[:,1,1] = evals[:,1]

        #Current Rotation

        V_dag = np.conj(evecs.transpose(0,2,1))
        J_eigen = V_dag @ J_model @ evecs

        #RWA isolation
        
        self.J_plus = np.zeros(J_eigen)
        self.J_minus= np.zeros(J_eigen)

        self.J_plus[:,1,0] = J_eigen[:,1,0]
        self.J_minus[:,0,1] = J_eigen[:,0,1]

        n_bands = H_model.shape[-1]
        for i in range(n_bands):
            for j in range(n_bands):
                if j > i: # Condition de montée en énergie
                    self.J_plus[:, j, i] = J_eigen[:, j, i]


    def set_dissipation(self, c_ops_list):
        """
        c_ops_list: Liste de tuples [(Opérateur_Qobj_ou_Matrice, taux), ...]
        """
        self.c_ops = c_ops_list

    # --- Utilitaires algébriques internes ---
    def _spre(self, A):
        return np.kron(np.eye(A.shape[1]), A)
    
    def _spost(self, A):
        return np.kron(A.T, np.eye(A.shape[0]))

    def _get_lindblad(self, C, gamma):
        C_dag = np.conj(C.transpose(0, 2, 1))
        C_dag_C = C_dag @ C
        return gamma * (self._spre(C) @ self._spost(C_dag) - 
                        0.5 * self._spre(C_dag_C) - 
                        0.5 * self._spost(C_dag_C)) 
    

    def get_thermal_state_numpy(self):
    # np.linalg.eigh fonctionne nativement sur un batch de matrices (N, 2, 2)
        evals, evecs = np.linalg.eigh(self.H_eigen)
        
        # Distribution de Fermi-Dirac (taille N, 2)
        f_occ = 1.0 / (np.exp((evals - self.mu) / self.T) + 1.0)
        
        rho_eq = np.zeros_like(self.H_eigen)
        # Construction de rho_eq par produit externe pour chaque k
        for i in range(2):
            vec = evecs[:, :, i:i+1] # Vecteur propre i
            vec_dag = np.conj(vec.transpose(0, 2, 1))
            rho_eq += f_occ[:, i:i+1, np.newaxis] * (vec @ vec_dag)
            
        return rho_eq

    def _get_L_eff(self, w):
        # Liouvillien cohérent -i[H, rho]
        L = -1j * (self._spre(self.H_eigen) - self._spost(self.H_eigen))
        # Ajout Lindblad
        for C, gamma in self.c_ops:
            L += self._get_lindblad(C, gamma)
        return L



    def calc_rephasing(self, w3, w1, tau2):


        t1, t2, T, mu = self.params["t1"], self.params["t2"], self.params["T"], self.params["mu"]


        rho_eq = self.get_thermal_state_numpy()





        # 1. Construction des propagateurs (Résolvantes)
        G1 = np.linalg.inv(1j * (w1 - self.H_eigen) + self.eta/2 * np.eye(self.H_eigen.shape[1])) # (Simpli.)
        G3 = np.linalg.inv(1j * (w3 - self.H_eigen) + self.eta/2 * np.eye(self.H_eigen.shape[1]))
        
        # 2. Construction du Liouvillien pour tau2 (incluant Lindblad)
        L_0 = self._get_L_eff(w=0) 
        
        # Propagation de la cohérence pendant tau2 (superopérateur)
        # On utilise exp(L * tau2)
        from scipy.linalg import expm
        Prop_tau2 = expm(L_0 * tau2)
        
        # 3. Calcul du chemin de Feynman
        # Ceci est une structure simplifiée basée sur tes diagrammes :
        # S3 ~ Tr(J_minus * G3 * J_plus * Prop_tau2 * J_plus * G1 * J_minus * rho_eq)
        # Note: Dans ton code, tu effectues ces produits matriciels point par point sur les k
        
        # Exemple de structure pour un terme (à adapter selon ta convention de signe)
        S3 = np.einsum('nij,njk,nkl,nlm,nmi->n', self.J_minus, G3, self.J_plus, Prop_tau2, self.J_plus @ self.J_minus)
        return S3





def calc_rephasing_S3_vectorized(k_array, w3, w1, tau2,params, c_ops=None):
    t1, t2 = params["t1"], params["t2"]
    eta, T = params["Eta"], params["T"]
    wLO, lam, mu = params["omegaLO"], params["lambda"], params["mu"]

    # 1. Génération des blocs dans la base des ÉNERGIES (RWA)
    # On récupère l'Hamiltonien diagonal et les opérateurs RWA purs
    H_eigen, J_plus, J_minus = get_H_and_Js_RWA_numpy(k_array, t1, t2)
    rho_eq = get_thermal_state_numpy(H_eigen, T, mu)

    # 2. Superopérateurs (Respectant la RWA)
    # J_plus ne fait QUE monter en énergie, J_minus ne fait QUE descendre
    JL_plus = spre_numpy(J_plus)
    JR_plus = spost_numpy(J_plus)
    JR_minus = spost_numpy(J_minus)
    
    # Opérateur de trace finale (mesure du dipôle / courant total)
    J_total = J_plus + J_minus
    JL_out = spre_numpy(J_total)

    # 3. Propagateurs (Utilisent H_eigen, la dynamique est désormais diagonale)
    L_w1 = get_L_eff_numpy(H_eigen, w1, wLO, lam, eta, T,c_ops)
    L_w3 = get_L_eff_numpy(H_eigen, w3, wLO, lam, eta, T, c_ops)
    L_0  = get_L_eff_numpy(H_eigen, 0.0, wLO, lam, eta, T, c_ops)
    G1, G2, G3 = get_green_functions_numpy(L_w1, L_0, L_w3, tau2, w1, w3, eta)

    # 4. Aplatissement de la matrice de densité (équivalent 'F' de numpy/QuTiP)
    N = len(k_array)
    rho_vec = np.zeros((N, 4, 1), dtype=complex)
    rho_vec[:, 0, 0] = rho_eq[:, 0, 0]
    rho_vec[:, 1, 0] = rho_eq[:, 1, 0]
    rho_vec[:, 2, 0] = rho_eq[:, 0, 1]
    rho_vec[:, 3, 0] = rho_eq[:, 1, 1]

    # ========================================================================
    # 5. Chemins de Feynman (Spectroscopie de Rephasage : -k1, +k2, +k3)
    # L'ordre d'application doit être rigoureux pour respecter la RWA.
    # ========================================================================
    
    # GSB : Absorbe -k1 sur le bra (JR_minus), interagit +k2 sur le bra (JR_plus), sonde +k3 sur le ket (JL_plus)
    path_GSB = G3 @ (JL_plus @ (G2 @ (JR_plus @ (G1 @ (JR_minus @ rho_vec)))))

    # SE : Absorbe -k1 sur le bra (JR_minus), interagit +k2 sur le ket (JL_plus), sonde +k3 sur le bra (JR_plus)
    path_SE  = G3 @ (JR_plus @ (G2 @ (JL_plus @ (G1 @ (JR_minus @ rho_vec)))))

    # ESA : Absorbe -k1 sur le bra (JR_minus), interagit +k2 sur le ket (JL_plus), interagit +k3 sur le ket (JL_plus)
    path_ESA = G3 @ (JL_plus @ (G2 @ (JL_plus @ (G1 @ (JR_minus @ rho_vec)))))

    # 6. Traces finales
    I_vec = np.zeros((N, 1, 4), dtype=complex)
    I_vec[:, 0, 0] = 1.0 
    I_vec[:, 0, 3] = 1.0

    tr_GSB = (I_vec @ (JL_out @ path_GSB)).reshape(-1)
    tr_SE  = (I_vec @ (JL_out @ path_SE)).reshape(-1)
    tr_ESA = (I_vec @ (JL_out @ path_ESA)).reshape(-1)

    # Addition des signaux (Soustraction conventionnelle de l'ESA)
    total_trace = -1j * (tr_GSB + tr_SE - tr_ESA)

    return total_trace



    def calc_unrephasing(self, w3, w1, tau2):
        """Calcul de la séquence +k1, -k2, +k3"""
        pass

    def generate_2D_spectra(self, w_list, tau2, dk):
        """
        Boucle principale :
        1. Itère sur w1 et w3
        2. Appelle calc_rephasing/unrephasing
        3. Somme sur les k-points (Intégration Brillouin)
        """
        n_w = len(w_list)
        S3_rephasing_grid = np.zeros((n_w, n_w), dtype=complex)
        
        for i, w1 in enumerate(w_list):
            for j, w3 in enumerate(w_list):
                # Calcul vecteur k (le résultat est de taille N)
                vec_k = self.calc_rephasing(w3, w1, tau2)
                
                # Somme sur k (Intégration)
                S3_rephasing_grid[i, j] = np.sum(vec_k) * dk / (2 * np.pi)
                
        return S3_rephasing_grid                 

'''        
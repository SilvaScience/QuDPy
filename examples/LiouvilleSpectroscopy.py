import numpy as np
import matplotlib.pyplot as plt
import time

###################################################################################
#################             Author: Mathieu Desmarais            ################
#################                Date: 10-06-2026                  ################
#################             2D spectroscopy solver               ################
###################################################################################



class LiouvilleSpectroscopySolver:
    def __init__(self, params):
        """
        Universal Liouville space solver for 2D spectroscopy.
        params : Dictionary containing 'Eta' (broadening) and 'T' (temperature).
        """
        self.params = params
        self.eta = params.get("Eta", 0.05)
        self.T = params.get("T", 0.01)
        
        # Model state variables (populated by feed_model)
        self.H_eigen = None
        self.J_plus = None
        self.J_minus = None

        self.c_ops = []
        self.c_ops_eigen = []



        self.dim = None
        self.N_k = None

        self.profiling = {
            "rephasing_total": 0.0,
            "unrephasing_total": 0.0,
            "resolvent_inversions": 0.0, # Temps passé dans np.linalg (G1, G2, G3)
            "feynman_pathways": 0.0,     # Temps passé dans les multiplications (@)
            "k_integration": 0.0,        # Temps passé dans np.sum
            "total_scan": 0.0
        }

    def feed_model(self, H_model, J_model, mu_optical):
        """
        Receives the Hamiltonian and Current in the site basis (N_k, d, d).
        Diagonalizes the system and universally separates the operators according to the RWA.
        """
        self.N_k, self.dim, _ = H_model.shape
        d = self.dim
        
        # 1. Diagonalization of the Hamiltonian over all k-points
        evals, evecs = np.linalg.eigh(H_model)
        
        # Proper construction of H_eigen (diagonal)
        self.H_eigen = np.zeros_like(H_model)
        for i in range(d):
            self.H_eigen[:, i, i] = evals[:, i]
            
        # 2. Rotation of the current operator into the eigenstate basis
        V_dag = np.conj(evecs.transpose(0, 2, 1))
        J_eigen = V_dag @ J_model @ evecs
        #    Rotation of the dipole operators

        mu_eigen = V_dag @ mu_optical @ evecs


        self.c_ops_eigen = []
        for self.c_ops, gamma in self.c_ops:
            # On étend C_local sur N_k s'il est statique
            C_k = np.tile(self.c_ops, (self.N_k, 1, 1))
            
            # Rotation dans la base propre
            C_eigen = V_dag @ C_k @ evecs
            self.c_ops_eigen.append((C_eigen, gamma))
        
        # 3. Universal application of the RWA (independent of the number of bands)
        self.J_plus = np.zeros_like(J_eigen)
        self.J_minus = np.zeros_like(J_eigen)
        
        # j > i implies an upward transition in energy (absorption)
        for i in range(d):
            for j in range(d):
                if j > i:
                    self.J_plus[:, j, i] = J_eigen[:, j, i]
                elif j < i:
                    self.J_minus[:, j, i] = J_eigen[:, j, i]

    def set_dissipation(self, c_ops_list):
        """
        Defines the Lindblad jump operators.
        c_ops_list : List of tuples [(Matrix_N_k_d_d, gamma_rate), ...]
        """
        self.c_ops = c_ops_list

    # ========================================================================
    # VECTORIZED LIOUVILLE ALGEBRA (BATCHED OVER K)
    # ========================================================================
    def _spre(self, A):
        """Left-acting superoperator: I_d \otimes A"""
        I = np.eye(self.dim)
        res = np.einsum('ij,nkl->nikjl', I, A)
        return res.reshape(self.N_k, self.dim**2, self.dim**2)
    
    def _spost(self, A):
        """Right-acting superoperator: A^T \otimes I_d"""
        I = np.eye(self.dim)
        res = np.einsum('nji,kl->nikjl', A, I)
        return res.reshape(self.N_k, self.dim**2, self.dim**2)

    def _get_lindblad(self, C, gamma):
        """Computes the full Lindblad dissipation matrix (N_k, d^2, d^2)"""
        C_dag = np.conj(C.transpose(0, 2, 1))
        C_dag_C = C_dag @ C
        return gamma * (self._spre(C) @ self._spost(C_dag) - 
                        0.5 * self._spre(C_dag_C) - 
                        0.5 * self._spost(C_dag_C))

    def _get_L_eff(self, w_probe):
        L = (self._spre(self.H_eigen) - self._spost(self.H_eigen))
        for C_eigen, gamma in self.c_ops_eigen: # <- Utiliser C_eigen ici
            L += 1j * self._get_lindblad(C_eigen, gamma)
        return L

    def _get_thermal_state(self):
        """Computes the vectorized initial thermal equilibrium state (N_k, d^2, 1)"""
        d = self.dim
        mu = self.params.get("mu", 0.0)
        
        if self.T > 0:
            beta = 1.0 / self.T
            energies = np.array([self.H_eigen[:, i, i] for i in range(d)]).T
            exp_factors = np.exp(-beta * (energies - mu))
            rho_diag = exp_factors / np.sum(exp_factors, axis=1, keepdims=True)
        else:
            rho_diag = np.zeros((self.N_k, d))
            rho_diag[:, 0] = 1.0 # Everything in the ground state at T=0
            
        rho_vec = np.zeros((self.N_k, d**2, 1), dtype=complex)
        for i in range(d):
            rho_vec[:, i * d + i, 0] = rho_diag[:, i]
        return rho_vec

    # ========================================================================
    # CALCULATION OF SPECTRA AND FEYNMAN PATHWAYS
    # ========================================================================
    def calc_rephasing(self, w3, w1, tau2):
        """Calculation of rephasing diagrams (-k1, +k2, +k3)"""
        d2 = self.dim**2
        I_super = np.eye(d2)
        rho_vec = self._get_thermal_state()
        
        # Initialization of interaction superoperators
        JL_plus = self._spre(self.J_plus)
        JR_plus = self._spost(self.J_plus)
        JR_minus = self._spost(self.J_minus)
        JL_out = self._spre(self.J_plus + self.J_minus)
        
        # Resolvents (G1 at -w1 for rephasing)
        L_w1 = self._get_L_eff(w1)
        L_w3 = self._get_L_eff(w3)
        L_0  = self._get_L_eff(0.0)
        

        t_inv_start = time.time()
        G1 = np.linalg.inv((w1 + 1j * self.eta) * I_super - L_w1)
        G3 = np.linalg.inv((w3 + 1j * self.eta) * I_super - L_w3)
        
        # Vectorized matrix exponential via diagonalization for tau2
        evals, evecs = np.linalg.eig(-1j * L_0 * tau2)
        G2 = (evecs * np.exp(evals)[:, np.newaxis, :]) @ np.linalg.inv(evecs)
        self.profiling["resolvent_inversions"] += time.time() - t_inv_start
        

        t_path_start = time.time()
        # Evaluation of Feynman pathways
        path_GSB = G3 @ (JL_plus @ (G2 @ (JR_plus @ (G1 @ (JR_minus @ rho_vec)))))
        path_SE  = G3 @ (JR_plus @ (G2 @ (JL_plus @ (G1 @ (JR_minus @ rho_vec)))))
        path_ESA = G3 @ (JL_plus @ (G2 @ (JL_plus @ (G1 @ (JR_minus @ rho_vec)))))
        
        # Trace operator
        I_vec = np.zeros((self.N_k, 1, d2), dtype=complex)
        for i in range(self.dim):
            I_vec[:, 0, i * self.dim + i] = 1.0
            
        tr_GSB = (I_vec @ (JL_out @ path_GSB)).reshape(-1)
        tr_SE  = (I_vec @ (JL_out @ path_SE)).reshape(-1)
        tr_ESA = (I_vec @ (JL_out @ path_ESA)).reshape(-1)

        self.profiling["feynman_pathways"] += time.time() - t_path_start
        
        return -1j * (tr_GSB + tr_SE - tr_ESA)

    def calc_unrephasing(self, w3, w1, tau2):
        """Calculation of non-rephasing diagrams (+k1, -k2, +k3)"""
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

        t_inv_start = time.time()
        # G1 is at +w1 for non-rephasing
        G1 = np.linalg.inv((w1 + 1j * self.eta) * I_super - L_w1)
        G3 = np.linalg.inv((w3 + 1j * self.eta) * I_super - L_w3)
        self.profiling["resolvent_inversions"] += time.time() - t_inv_start

        evals, evecs = np.linalg.eig(-1j * L_0 * tau2)
        G2 = (evecs * np.exp(evals)[:, np.newaxis, :]) @ np.linalg.inv(evecs)
        
        t_path_start = time.time()
        # Non-rephasing sequence
        path_GSB = G3 @ (JL_plus @ (G2 @ (JL_minus @ (G1 @ (JL_plus @ rho_vec)))))
        path_SE  = G3 @ (JR_plus @ (G2 @ (JR_minus @ (G1 @ (JL_plus @ rho_vec)))))
        path_ESA = G3 @ (JL_plus @ (G2 @ (JR_minus @ (G1 @ (JL_plus @ rho_vec)))))
        
        I_vec = np.zeros((self.N_k, 1, d2), dtype=complex)
        for i in range(self.dim):
            I_vec[:, 0, i * self.dim + i] = 1.0
            
        tr_GSB = (I_vec @ (JL_out @ path_GSB)).reshape(-1)
        tr_SE  = (I_vec @ (JL_out @ path_SE)).reshape(-1)
        tr_ESA = (I_vec @ (JL_out @ path_ESA)).reshape(-1)

        self.profiling["feynman_pathways"] += time.time() - t_path_start
        
        return -1j * (tr_GSB + tr_SE - tr_ESA)

    def generate_2D_spectra(self, w_list, tau2, k_array):
        """
        Scans the w1 and w3 frequency grid.
        Performs the sum over the Brillouin zone (integration over k).
        Returns: A dictionary containing the complex 2D response matrices.
        """
        n_w = len(w_list)
        dk = (k_array[-1] - k_array[0]) / len(k_array)
        
        S3_reph = np.zeros((n_w, n_w), dtype=complex)
        S3_unreph = np.zeros((n_w, n_w), dtype=complex)

        print(f"Début du balayage 2D : Grille de {n_w}x{n_w} points...")
        t_scan_start = time.time()
        
        for i, w1 in enumerate(w_list):
            for j, w3 in enumerate(w_list):
                # Simultaneous calculation for all k-points
                t0 = time.time()
                vec_reph = self.calc_rephasing(w3, w1, tau2)
                self.profiling["rephasing_total"] += time.time() - t0
                
                t0 = time.time()
                vec_unreph = self.calc_unrephasing(w3, w1, tau2)
                self.profiling["unrephasing_total"] += time.time() - t0
                
                # Numerical trapezoidal integration / sum over the Brillouin zone
                t0 = time.time()
                S3_reph[j, i] = np.sum(vec_reph) * dk / (2 * np.pi)
                S3_unreph[j, i] = np.sum(vec_unreph) * dk / (2 * np.pi)
                self.profiling["k_integration"] += time.time() - t0
                
        self.profiling["total_scan"] = time.time() - t_scan_start 
        self._print_profiling_report()       
                
        return {
            "rephasing": S3_reph,
            "unrephasing": S3_unreph,
            "absorptive": S3_reph + S3_unreph
        }
    
    def _print_profiling_report(self):
        """Print a report of the execution time in % and seconds"""
        total = self.profiling["total_scan"]
        print("\n" + "="*50)
        print(" Report - LIOUVILLE SPECTROSCOPY ")
        print("="*50)
        print(f"Total execution times: {total:.2f} seconds")
        print("-" * 50)
        
        # Détail des sous-catégories
        categories = {
            "Rephasing global": self.profiling["rephasing_total"],
            "Unrephasing global": self.profiling["unrephasing_total"],
            "-> Inversions (G1, G2, G3)": self.profiling["resolvent_inversions"],
            "-> Contractions Feynman": self.profiling["feynman_pathways"],
            "Intégration (Sum k)": self.profiling["k_integration"]
        }
        
        for name, timing in categories.items():
            percent = (timing / total) * 100 if total > 0 else 0
            print(f"{name:<28} : {timing:>6.2f} s  ({percent:>5.1f}%)")
        print("="*50 + "\n")


class SpectroscopyPlotter:
    def __init__(self, w_list): 
        self.w_list = w_list

    def plot_spectrum(self, S3_topo, S3_triv, levels=60, figsize=(12, 14), save_path=None):
        """
        Displays a 3x2 figure with the 2D spectra (Real, Imag, Abs)
        for the topological and trivial phases.
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
        """Private method to avoid repeating the contourf code"""
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
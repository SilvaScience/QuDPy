import numpy as np
import matplotlib.pyplot as plt

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
        self.dim = None
        self.N_k = None

    def feed_model(self, H_model, J_model):
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
        """Generates the total effective Liouvillian for a given frequency"""
        # Coherent part: [H, \rho]
        L = (self._spre(self.H_eigen) - self._spost(self.H_eigen))
        # Addition of Lindblad relaxation
        for C, gamma in self.c_ops:
            L += 1j * self._get_lindblad(C, gamma)
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
        
        G1 = np.linalg.inv((-w1 + 1j * self.eta) * I_super - L_w1)
        G3 = np.linalg.inv((w3 + 1j * self.eta) * I_super - L_w3)
        
        # Vectorized matrix exponential via diagonalization for tau2
        evals, evecs = np.linalg.eig(-1j * L_0 * tau2)
        G2 = (evecs * np.exp(evals)[:, np.newaxis, :]) @ np.linalg.inv(evecs)
        
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
        
        # G1 is at +w1 for non-rephasing
        G1 = np.linalg.inv((w1 + 1j * self.eta) * I_super - L_w1)
        G3 = np.linalg.inv((w3 + 1j * self.eta) * I_super - L_w3)

        evals, evecs = np.linalg.eig(-1j * L_0 * tau2)
        G2 = (evecs * np.exp(evals)[:, np.newaxis, :]) @ np.linalg.inv(evecs)
        
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
        
        for i, w1 in enumerate(w_list):
            for j, w3 in enumerate(w_list):
                # Simultaneous calculation for all k-points
                vec_reph = self.calc_rephasing(-w3, w1, tau2)
                vec_unreph = self.calc_unrephasing(w3, w1, tau2)
                
                # Numerical trapezoidal integration / sum over the Brillouin zone
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
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm, expm_multiply, factorized, gmres
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


_PROCESS_SOLVER = None


def _init_process_solver(solver):
    """Store one solver copy per process worker."""
    global _PROCESS_SOLVER
    _PROCESS_SOLVER = solver


def _calc_w3_block_process(block, w_list, tau2, integration_factor):
    """ProcessPool worker entry point using the process-local solver."""
    if _PROCESS_SOLVER is None:
        raise RuntimeError("Process worker was not initialized with a solver.")
    return _PROCESS_SOLVER._calc_w3_block(block, w_list, tau2, integration_factor)


def _calc_w3_block_joblib(solver, block, w_list, tau2, integration_factor):
    """Joblib worker entry point for process-style backends."""
    return solver._calc_w3_block(block, w_list, tau2, integration_factor)

###################################################################################
#################             Author: Mathieu Desmarais            ################
#################                Date: 10-06-2026                  ################
#################             2D spectroscopy solver               ################
###################################################################################


class LiouvilleSpectroscopySolver:
    def __init__(self, params):
        """
        Universal Liouville-space solver for 2D spectroscopy.

        Parameters
        ----------
        params : dict
            Expected keys include:
            - Eta: spectral broadening
            - T: temperature
            - mu: chemical potential
            - cache_resolvents: whether sparse LU factorizations are cached
            - max_resolvent_cache: optional maximum number of cached factorizations
        """
        self.params = params
        self.eta = params.get("Eta", 0.05)
        self.T = params.get("T", 0.01)
        self.rwa_tol = params.get("rwa_tol", 1e-6)
        self.cache_resolvents = params.get("cache_resolvents", True)
        self.max_resolvent_cache = params.get("max_resolvent_cache", None)
        self.backend = params.get("backend", "auto")
        self.dense_liouville_cutoff = params.get("dense_liouville_cutoff", 512)
        self.sparse_solver = params.get("sparse_solver", "auto")
        self.sparse_gmres_rtol = params.get("sparse_gmres_rtol", 1e-8)
        self.sparse_gmres_atol = params.get("sparse_gmres_atol", 0.0)
        self.sparse_gmres_maxiter = params.get("sparse_gmres_maxiter", None)
        self.cache_sparse_tau2 = params.get("cache_sparse_tau2", False)
        self.parallel_backend = params.get("parallel_backend", "threading")
        self.parallel_block_size = params.get("parallel_block_size", None)
        self.blas_threads = params.get("blas_threads", None)
        self.n_jobs = params.get("n_jobs", -1)
        self._active_backend = None
        self._sparse_direct_failed = False

        self.H_eigen = None
        self.energies = None
        self.eigenvectors = None
        self.J_plus = None
        self.J_minus = None

        self.c_ops = []
        self.c_ops_eigen = []

        self.dim = None
        self.N_k = None

        self._I_d_sp = None
        self._I_super_sp = None
        self._trace_vec = None
        self._rho_eq = None
        self._JL_plus = []
        self._JR_plus = []
        self._JL_minus = []
        self._JR_minus = []
        self._JL_out = []
        self._L_eff_sp = []
        self._resolvent_cache = OrderedDict()
        self._sparse_tau2_cache = OrderedDict()
        self._I_super_dense = None
        self._rho_eq_dense = None
        self._trace_vec_dense = None
        self._JL_plus_dense = None
        self._JR_plus_dense = None
        self._JL_minus_dense = None
        self._JR_minus_dense = None
        self._JL_out_dense = None
        self._L_eff_dense = None
        self._dense_resolvent_cache = OrderedDict()
        self._dense_tau2_cache = OrderedDict()

    # ========================================================================
    # Parallelisation method
    # ========================================================================

    def _calc_w3_column(self, i, w3, w_list, tau2, integration_factor):
        """
        Calcul a complete omega column of the 2D spectra for a w3 fix frequency.
        This function is exectued by a independant worker.
        """

        n_w = len(w_list)
        col_reph= np.zeros(n_w, dtype=np.complex128)
        col_unreph = np.zeros(n_w, dtype=np.complex128)

        for j, w1 in enumerate(w_list):
            vect_reph = self.calc_rephasing(w3, w1, tau2)
            vect_unreph = self.calc_unrephasing(w3, w1, tau2)

            col_reph[j] = np.sum(vect_reph) * integration_factor
            col_unreph[j] = np.sum(vect_unreph) * integration_factor

        return i, col_reph, col_unreph

    def _calc_w3_block(self, block, w_list, tau2, integration_factor):
        """Calculate a block of omega_3 columns."""
        return [
            self._calc_w3_column(i, w_list[i], w_list, tau2, integration_factor)
            for i in block
        ]

    def _effective_n_jobs(self, n_jobs):
        """Normalize joblib-style worker counts."""
        cpu_count = os.cpu_count() or 1
        if n_jobs is None:
            return 1
        if n_jobs < 0:
            return max(1, cpu_count + 1 + n_jobs)
        return max(1, int(n_jobs))

    def _make_w3_blocks(self, n_w, n_jobs, block_size):
        """Split omega_3 column indices into backend tasks."""
        if block_size is None:
            block_size = self.parallel_block_size
        if block_size is None:
            target_blocks = max(1, 4 * max(1, n_jobs))
            block_size = max(1, math.ceil(n_w / target_blocks))
        block_size = max(1, int(block_size))
        return [
            list(range(start, min(start + block_size, n_w)))
            for start in range(0, n_w, block_size)
        ]

    def _parallel_context(self, blas_threads):
        """Limit nested BLAS/OpenMP threads while the outer omega loop runs."""
        if blas_threads is None:
            blas_threads = self.blas_threads
        if blas_threads is None or threadpool_limits is None:
            return nullcontext()
        return threadpool_limits(limits=blas_threads)

    def _run_w3_blocks(
        self,
        blocks,
        w_list,
        tau2,
        integration_factor,
        parallel_backend,
        n_jobs,
    ):
        """Run omega_3 blocks using the selected parallel backend."""
        if parallel_backend in {"serial", None} or n_jobs == 1:
            return [
                item
                for block in blocks
                for item in self._calc_w3_block(block, w_list, tau2, integration_factor)
            ]

        if parallel_backend == "threading":
            nested = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(self._calc_w3_block)(
                    block, w_list, tau2, integration_factor
                )
                for block in blocks
            )
        elif parallel_backend in {"loky", "multiprocessing"}:
            nested = Parallel(
                n_jobs=n_jobs,
                backend=parallel_backend,
                max_nbytes="10M",
                mmap_mode="r",
            )(
                delayed(_calc_w3_block_joblib)(
                    self, block, w_list, tau2, integration_factor
                )
                for block in blocks
            )
        elif parallel_backend in {"process", "processpool"}:
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_process_solver,
                initargs=(self,),
            ) as pool:
                futures = [
                    pool.submit(
                        _calc_w3_block_process,
                        block,
                        w_list,
                        tau2,
                        integration_factor,
                    )
                    for block in blocks
                ]
                nested = [future.result() for future in futures]
        else:
            raise ValueError(
                "parallel_backend must be one of: "
                "'serial', 'threading', 'loky', 'multiprocessing', or 'process'"
            )

        return [item for block_result in nested for item in block_result]


    # ========================================================================
    # INPUT NORMALIZATION
    # ========================================================================


    def _clean_gamma(self, gamma):
        """Return a real scalar Lindblad rate."""
        if gamma is None:
            return None

        gamma = np.asarray(gamma).item()
        gamma = np.real_if_close(gamma)
        if np.iscomplexobj(gamma):
            if abs(np.imag(gamma)) > 1e-12:
                raise ValueError(f"Lindblad rates must be real, got {gamma}")
            gamma = np.real(gamma)
        return float(gamma)

    def _as_k_stack(self, array, name):
        """
        Normalize a matrix-like object to shape (N_k, d, d).

        Accepted input shapes are:
        - (d, d)
        - (N_k, d, d)
        """
        array = np.asarray(array, dtype=np.complex128)

        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        elif array.ndim != 3:
            raise ValueError(
                f"{name} must have shape (d, d) or (N_k, d, d); got {array.shape}"
            )

        if array.shape[1] != array.shape[2]:
            raise ValueError(f"{name} must contain square matrices; got {array.shape}")

        return array

    def _broadcast_stack(self, array, name, N_k, dim):
        """Broadcast a single-k matrix stack and validate all dimensions."""
        if array.shape[1:] != (dim, dim):
            raise ValueError(
                f"{name} has matrix shape {array.shape[1:]}, expected {(dim, dim)}"
            )

        if array.shape[0] == N_k:
            return array

        if array.shape[0] == 1 and N_k > 1:
            return np.repeat(array, N_k, axis=0)

        raise ValueError(f"{name} has N_k={array.shape[0]}, expected {N_k}")

    def _split_c_ops(self, c_ops_raw):
        """
        Normalize collapse operators without deciding the final N_k yet.

        Each item can be either:
        - C
        - (C, gamma)
        """
        if c_ops_raw is None:
            return []

        c_ops_prepared = []
        for idx, item in enumerate(c_ops_raw):
            if isinstance(item, tuple) and len(item) == 2:
                C_raw, gamma = item
            else:
                C_raw, gamma = item, None

            C_raw = self._as_k_stack(C_raw, f"c_ops_raw[{idx}]")
            c_ops_prepared.append((C_raw, self._clean_gamma(gamma)))

        return c_ops_prepared

    def _prepare_model_inputs(self, H_model, interaction_op_array, c_ops_raw):
        """Validate and broadcast all model inputs to a common shape."""
        H_model = self._as_k_stack(H_model, "H_model")
        interaction_op_array = self._as_k_stack(
            interaction_op_array, "interaction_op_array"
        )
        c_ops_prepared = self._split_c_ops(c_ops_raw)

        dim = H_model.shape[1]
        candidate_N_k = [H_model.shape[0], interaction_op_array.shape[0]]
        candidate_N_k.extend(C_raw.shape[0] for C_raw, _ in c_ops_prepared)
        N_k = max(candidate_N_k)

        H_model = self._broadcast_stack(H_model, "H_model", N_k, dim)
        interaction_op_array = self._broadcast_stack(
            interaction_op_array, "interaction_op_array", N_k, dim
        )

        c_ops_broadcasted = []
        for idx, (C_raw, gamma) in enumerate(c_ops_prepared):
            C_raw = self._broadcast_stack(C_raw, f"c_ops_raw[{idx}]", N_k, dim)
            c_ops_broadcasted.append((C_raw, gamma))

        return H_model, interaction_op_array, c_ops_broadcasted

    # ========================================================================
    # MODEL LOADING
    # ========================================================================
    def feed_model(
        self,
        H_model,
        interaction_op_array,
        c_ops_raw=None,
        interaction_type="dipole",
    ):
        """
        Load a model in the site basis and prepare sparse Liouville operators.

        H_model and interaction_op_array must have shape (d, d) or (N_k, d, d).
        c_ops_raw may contain raw matrices or (matrix, gamma) tuples. If gammas
        are provided here, the dissipation is configured immediately.
        """
        H_model, interaction_op_array, c_ops_prepared = self._prepare_model_inputs(
            H_model, interaction_op_array, c_ops_raw
        )

        print(f"--- Model loading (interaction type: {interaction_type}) ---")

        self.N_k, self.dim, _ = H_model.shape
        self.energies = np.zeros((self.N_k, self.dim), dtype=float)
        self.eigenvectors = np.zeros(
            (self.N_k, self.dim, self.dim), dtype=np.complex128
        )
        self.H_eigen = np.zeros(
            (self.N_k, self.dim, self.dim), dtype=np.complex128
        )
        self.J_plus = np.zeros(
            (self.N_k, self.dim, self.dim), dtype=np.complex128
        )
        self.J_minus = np.zeros(
            (self.N_k, self.dim, self.dim), dtype=np.complex128
        )
        self.c_ops_eigen = [
            np.zeros((self.N_k, self.dim, self.dim), dtype=np.complex128)
            for _ in c_ops_prepared
        ]

        for i_k in range(self.N_k):
            evals, evecs = np.linalg.eigh(H_model[i_k])
            evals = np.real_if_close(evals).real

            self.energies[i_k] = evals
            self.eigenvectors[i_k] = evecs
            self.H_eigen[i_k] = np.diag(evals)

            U = evecs
            U_dag = U.conj().T

            O_eigen = U_dag @ interaction_op_array[i_k] @ U

            if interaction_type == "dipole":
                delta_E = evals[:, np.newaxis] - evals[np.newaxis, :]
                self.J_plus[i_k] = np.where(delta_E > self.rwa_tol, O_eigen, 0.0)
                self.J_minus[i_k] = np.where(delta_E < -self.rwa_tol, O_eigen, 0.0)
            elif interaction_type == "current":
                self.J_plus[i_k] = np.tril(O_eigen, k=-1)
                self.J_minus[i_k] = np.triu(O_eigen, k=1)
            else:
                raise ValueError("interaction_type must be 'dipole' or 'current'")

            for idx, (C_raw, _) in enumerate(c_ops_prepared):
                self.c_ops_eigen[idx][i_k] = U_dag @ C_raw[i_k] @ U

        gammas = [gamma for _, gamma in c_ops_prepared]
        if any(gamma is not None for gamma in gammas):
            if any(gamma is None for gamma in gammas):
                raise ValueError(
                    "Either provide a gamma for every collapse operator or call "
                    "set_dissipation() after feed_model()."
                )
            self.c_ops = [
                (self.c_ops_eigen[idx], gamma) for idx, gamma in enumerate(gammas)
            ]
        else:
            self.c_ops = []

        self._build_liouville_backend()
        print("Model transformed to the eigenbasis.")
        print(f"Liouville backend ready: {self._active_backend}.")

    def set_dissipation(self, c_ops_list, basis="eigen"):
        """
        Define Lindblad jump operators.

        Parameters
        ----------
        c_ops_list : list
            List of (matrix_stack, gamma) pairs.
        basis : {"eigen", "site"}
            Use "eigen" when the operators are already projected. Use "site"
            to project them with the eigenvectors stored by feed_model().
        """
        if self.N_k is None or self.dim is None:
            raise RuntimeError("Call feed_model() before set_dissipation().")

        if basis not in {"eigen", "site"}:
            raise ValueError("basis must be either 'eigen' or 'site'")

        c_ops_eigen = []
        c_ops_with_gamma = []

        for idx, item in enumerate(c_ops_list):
            if not (isinstance(item, tuple) and len(item) == 2):
                raise ValueError(
                    "set_dissipation expects a list of (matrix_stack, gamma) pairs."
                )

            C_raw, gamma = item
            gamma = self._clean_gamma(gamma)
            C_raw = self._as_k_stack(C_raw, f"c_ops_list[{idx}]")
            C_raw = self._broadcast_stack(
                C_raw, f"c_ops_list[{idx}]", self.N_k, self.dim
            )

            if basis == "site":
                C_eigen = np.zeros_like(C_raw, dtype=np.complex128)
                for i_k in range(self.N_k):
                    U = self.eigenvectors[i_k]
                    C_eigen[i_k] = U.conj().T @ C_raw[i_k] @ U
            else:
                C_eigen = C_raw

            c_ops_eigen.append(C_eigen)
            c_ops_with_gamma.append((C_eigen, gamma))

        self.c_ops_eigen = c_ops_eigen
        self.c_ops = c_ops_with_gamma
        self._build_liouville_backend()
        print("Dissipation operators updated.")

    # ========================================================================
    # BACKEND SELECTION
    # ========================================================================
    def _select_backend(self):
        """Choose the fastest Liouville backend for the current problem size."""
        if self.backend not in {"auto", "dense", "sparse"}:
            raise ValueError("backend must be 'auto', 'dense', or 'sparse'")

        if self.backend in {"dense", "sparse"}:
            return self.backend

        d2 = self.dim**2
        if d2 <= self.dense_liouville_cutoff:
            return "dense"
        return "sparse"

    def _build_liouville_backend(self):
        """Build the active Liouville backend and clear stale caches."""
        self._active_backend = self._select_backend()
        self._sparse_direct_failed = False
        self._dense_resolvent_cache.clear()
        self._dense_tau2_cache.clear()
        self._resolvent_cache.clear()
        self._sparse_tau2_cache.clear()

        if self._active_backend == "dense":
            self._build_dense_liouville()
        else:
            self._build_sparse_liouville()

    # ========================================================================
    # DENSE VECTORIZED LIOUVILLE ALGEBRA
    # ========================================================================
    def _spre_dense(self, A):
        """Left-acting dense superoperator: I_d kron A."""
        I = np.eye(self.dim, dtype=np.complex128)
        res = np.einsum("ij,nkl->nikjl", I, A)
        return res.reshape(self.N_k, self.dim**2, self.dim**2)

    def _spost_dense(self, A):
        """Right-acting dense superoperator: A.T kron I_d."""
        I = np.eye(self.dim, dtype=np.complex128)
        res = np.einsum("nji,kl->nikjl", A, I)
        return res.reshape(self.N_k, self.dim**2, self.dim**2)

    def _get_lindblad_dense(self, C, gamma):
        """Build one dense Lindblad dissipator for every k-point."""
        C_dag = np.conj(C.transpose(0, 2, 1))
        C_dag_C = C_dag @ C
        return gamma * (
            self._spre_dense(C) @ self._spost_dense(C_dag)
            - 0.5 * self._spre_dense(C_dag_C)
            - 0.5 * self._spost_dense(C_dag_C)
        )

    def _build_dense_liouville(self):
        """Precompute dense batched superoperators and the static Liouvillian."""
        d2 = self.dim**2
        self._I_super_dense = np.eye(d2, dtype=np.complex128)
        self._rho_eq_dense = self._get_thermal_state()

        self._JL_plus_dense = self._spre_dense(self.J_plus)
        self._JR_plus_dense = self._spost_dense(self.J_plus)
        self._JL_minus_dense = self._spre_dense(self.J_minus)
        self._JR_minus_dense = self._spost_dense(self.J_minus)
        self._JL_out_dense = self._spre_dense(self.J_plus + self.J_minus)

        self._trace_vec_dense = np.zeros((self.N_k, 1, d2), dtype=np.complex128)
        for i in range(self.dim):
            self._trace_vec_dense[:, 0, i * self.dim + i] = 1.0

        self._L_eff_dense = (
            self._spre_dense(self.H_eigen) - self._spost_dense(self.H_eigen)
        ).astype(np.complex128)

        for C_eigen, gamma in self.c_ops:
            self._L_eff_dense += 1j * self._get_lindblad_dense(C_eigen, gamma)

    def _get_dense_resolvent(self, w):
        """
        Return cached dense resolvents for all k-points.

        The Liouvillian does not depend on w in this implementation, so each
        frequency only needs one batched inverse per scan.
        """
        key = float(np.round(w, 12))
        cached = self._dense_resolvent_cache.get(key)
        if cached is not None:
            return cached

   
        A = (w + 1j * self.eta) * self._I_super_dense - self._L_eff_dense
        G = np.linalg.inv(A)
      

        self._dense_resolvent_cache[key] = G
        self._dense_resolvent_cache.move_to_end(key)
        if (
            self.max_resolvent_cache is not None
            and len(self._dense_resolvent_cache) > self.max_resolvent_cache
        ):
            self._dense_resolvent_cache.popitem(last=False)
        return G

    def _get_dense_tau2_propagator(self, tau2):
        """Return cached dense G2 = exp(-i L tau2) for all k-points."""
        key = float(np.round(tau2, 12))
        cached = self._dense_tau2_cache.get(key)
        if cached is not None:
            return cached

    
        evals, evecs = np.linalg.eig(-1j * self._L_eff_dense * tau2)
        G2 = (evecs * np.exp(evals)[:, np.newaxis, :]) @ np.linalg.inv(evecs)
      

        self._dense_tau2_cache[key] = G2
        return G2

    def _calc_rephasing_dense(self, w3, w1, tau2):
        """Compute rephasing diagrams with the dense batched backend."""
      

        G1 = self._get_dense_resolvent(w1)
        G2 = self._get_dense_tau2_propagator(tau2)
        G3 = self._get_dense_resolvent(w3)
        rho = self._rho_eq_dense

        path_gsb = G3 @ (
            self._JL_plus_dense
            @ (G2 @ (self._JR_plus_dense @ (G1 @ (self._JR_minus_dense @ rho))))
        )
        path_se = G3 @ (
            self._JR_plus_dense
            @ (G2 @ (self._JL_plus_dense @ (G1 @ (self._JR_minus_dense @ rho))))
        )
        path_esa = G3 @ (
            self._JL_plus_dense
            @ (G2 @ (self._JL_plus_dense @ (G1 @ (self._JR_minus_dense @ rho))))
        )

        tr_gsb = (self._trace_vec_dense @ (self._JL_out_dense @ path_gsb)).reshape(-1)
        tr_se = (self._trace_vec_dense @ (self._JL_out_dense @ path_se)).reshape(-1)
        tr_esa = (self._trace_vec_dense @ (self._JL_out_dense @ path_esa)).reshape(-1)

       
        return -1j * (tr_gsb + tr_se - tr_esa)

    def _calc_unrephasing_dense(self, w3, w1, tau2):
        """Compute non-rephasing diagrams with the dense batched backend."""
    

        G1 = self._get_dense_resolvent(w1)
        G2 = self._get_dense_tau2_propagator(tau2)
        G3 = self._get_dense_resolvent(w3)
        rho = self._rho_eq_dense

        path_gsb = G3 @ (
            self._JL_plus_dense
            @ (G2 @ (self._JL_minus_dense @ (G1 @ (self._JL_plus_dense @ rho))))
        )
        path_se = G3 @ (
            self._JR_plus_dense
            @ (G2 @ (self._JR_minus_dense @ (G1 @ (self._JL_plus_dense @ rho))))
        )
        path_esa = G3 @ (
            self._JL_plus_dense
            @ (G2 @ (self._JR_minus_dense @ (G1 @ (self._JL_plus_dense @ rho))))
        )

        tr_gsb = (self._trace_vec_dense @ (self._JL_out_dense @ path_gsb)).reshape(-1)
        tr_se = (self._trace_vec_dense @ (self._JL_out_dense @ path_se)).reshape(-1)
        tr_esa = (self._trace_vec_dense @ (self._JL_out_dense @ path_esa)).reshape(-1)

       
        return -1j * (tr_gsb + tr_se - tr_esa)

    # ========================================================================
    # SPARSE LIOUVILLE ALGEBRA
    # ========================================================================
    def _spre_sp(self, A):
        """Left-acting superoperator: I_d kron A."""
        return sp.kron(self._I_d_sp, A, format="csr")

    def _spost_sp(self, A):
        """Right-acting superoperator: A.T kron I_d."""
        return sp.kron(A.T, self._I_d_sp, format="csr")

    def _get_lindblad_sp(self, C, gamma):
        """Build one sparse Lindblad dissipator in Liouville space."""
        C_dag = C.getH()
        C_dag_C = C_dag @ C
        return gamma * (
            self._spre_sp(C) @ self._spost_sp(C_dag)
            - 0.5 * self._spre_sp(C_dag_C)
            - 0.5 * self._spost_sp(C_dag_C)
        )

    def _build_sparse_liouville(self):
        """Precompute sparse superoperators used by the response functions."""
        if self.N_k is None or self.dim is None:
            return

        d2 = self.dim**2
        self._I_d_sp = sp.identity(self.dim, dtype=np.complex128, format="csr")
        self._I_super_sp = sp.identity(d2, dtype=np.complex128, format="csr")

        self._trace_vec = np.zeros(d2, dtype=np.complex128)
        for i in range(self.dim):
            self._trace_vec[i * self.dim + i] = 1.0

        self._rho_eq = self._get_thermal_state().reshape(self.N_k, d2)

        self._JL_plus = []
        self._JR_plus = []
        self._JL_minus = []
        self._JR_minus = []
        self._JL_out = []
        self._L_eff_sp = []

        for i_k in range(self.N_k):
            H_k = sp.diags(
                self.energies[i_k], offsets=0, dtype=np.complex128, format="csr"
            )
            J_plus_k = sp.csr_matrix(self.J_plus[i_k])
            J_minus_k = sp.csr_matrix(self.J_minus[i_k])

            self._JL_plus.append(self._spre_sp(J_plus_k))
            self._JR_plus.append(self._spost_sp(J_plus_k))
            self._JL_minus.append(self._spre_sp(J_minus_k))
            self._JR_minus.append(self._spost_sp(J_minus_k))
            self._JL_out.append(self._spre_sp(J_plus_k + J_minus_k))

            L = (self._spre_sp(H_k) - self._spost_sp(H_k)).astype(np.complex128)

            for C_stack, gamma in self.c_ops:
                C_k = sp.csr_matrix(C_stack[i_k])
                L = L + 1j * self._get_lindblad_sp(C_k, gamma)

            self._L_eff_sp.append(L.tocsr())

        self._resolvent_cache.clear()

    def _solve_resolvent(self, i_k, w, rhs):
        """
        Apply the resolvent without building an explicit inverse matrix.

        This solves:
            ((w + i eta) I - L) x = rhs
        """
        key = (i_k, float(np.round(w, 12)))
        t0 = time.time()

        A = None
        use_direct = self.sparse_solver in {"auto", "direct"} and not self._sparse_direct_failed

        if use_direct:
            solve = self._resolvent_cache.get(key)
            if solve is None:
                try:
                    A = (
                        (w + 1j * self.eta) * self._I_super_sp
                        - self._L_eff_sp[i_k]
                    ).tocsc()
                    solve = factorized(A)
                except MemoryError:
                    if self.sparse_solver == "direct":
                        raise
                    self._sparse_direct_failed = True
                    self._resolvent_cache.clear()
                    solve = None

                if solve is not None and self.cache_resolvents:
                    self._resolvent_cache[key] = solve
                    self._resolvent_cache.move_to_end(key)
                    if (
                        self.max_resolvent_cache is not None
                        and len(self._resolvent_cache) > self.max_resolvent_cache
                    ):
                        self._resolvent_cache.popitem(last=False)

            if solve is not None:
                result = solve(rhs)
                
                return result

        if A is None:
            A = ((w + 1j * self.eta) * self._I_super_sp - self._L_eff_sp[i_k]).tocsr()

        result, info = gmres(
            A,
            rhs,
            rtol=self.sparse_gmres_rtol,
            atol=self.sparse_gmres_atol,
            maxiter=self.sparse_gmres_maxiter,
        )
        if info != 0:
            raise RuntimeError(
                f"GMRES did not converge for k={i_k}, w={w}. info={info}"
            )
       
        return result

    def _propagate_tau2(self, i_k, tau2, rhs):
        """Apply exp(-i L tau2) to a vector using a sparse exponential action."""
        if not self.cache_sparse_tau2:
            return expm_multiply((-1j * tau2) * self._L_eff_sp[i_k], rhs)

        key = (i_k, float(np.round(tau2, 12)))
        G2 = self._sparse_tau2_cache.get(key)
        if G2 is None:
            G2 = expm((-1j * tau2) * self._L_eff_sp[i_k]).tocsr()
            self._sparse_tau2_cache[key] = G2
        return G2 @ rhs

    def _trace_output(self, i_k, response_vec):
        """Apply the output interaction and take the Liouville trace."""
        return self._trace_vec @ (self._JL_out[i_k] @ response_vec)

    def _get_thermal_state(self):
        """Compute the vectorized initial thermal equilibrium state."""
        d = self.dim
        d2 = d**2
        kB = 8.6173e-5
        mu = self.params.get("mu", 0.0)

        if self.T > 0:
            beta = 1.0 / (kB * self.T)
            shifted = self.energies - mu
            shifted = shifted - np.min(shifted, axis=1, keepdims=True)
            weights = np.exp(-beta * shifted)
            rho_diag = weights / np.sum(weights, axis=1, keepdims=True)
        else:
            rho_diag = np.zeros((self.N_k, d), dtype=float)
            rho_diag[:, 0] = 1.0

        rho_vec = np.zeros((self.N_k, d2, 1), dtype=np.complex128)
        for i in range(d):
            rho_vec[:, i * d + i, 0] = rho_diag[:, i]

        return rho_vec

    # ========================================================================
    # RESPONSE FUNCTIONS
    # ========================================================================
    def calc_rephasing(self, w3, w1, tau2):
        """Compute rephasing diagrams (-k1, +k2, +k3)."""
        if self._active_backend == "dense":
            return self._calc_rephasing_dense(w3, w1, tau2)

        if not self._L_eff_sp:
            raise RuntimeError("Call feed_model() before calc_rephasing().")

 
        response = np.zeros(self.N_k, dtype=np.complex128)

        for i_k in range(self.N_k):
            rho = self._rho_eq[i_k]

            v1 = self._JR_minus[i_k] @ rho
            v1 = self._solve_resolvent(i_k, w1, v1)

            mid_gsb = self._JR_plus[i_k] @ v1
            mid_gsb = self._propagate_tau2(i_k, tau2, mid_gsb)
            path_gsb = self._JL_plus[i_k] @ mid_gsb
            path_gsb = self._solve_resolvent(i_k, w3, path_gsb)

            mid_se_esa = self._JL_plus[i_k] @ v1
            mid_se_esa = self._propagate_tau2(i_k, tau2, mid_se_esa)

            path_se = self._JR_plus[i_k] @ mid_se_esa
            path_se = self._solve_resolvent(i_k, w3, path_se)

            path_esa = self._JL_plus[i_k] @ mid_se_esa
            path_esa = self._solve_resolvent(i_k, w3, path_esa)

            tr_gsb = self._trace_output(i_k, path_gsb)
            tr_se = self._trace_output(i_k, path_se)
            tr_esa = self._trace_output(i_k, path_esa)

            response[i_k] = -1j * (tr_gsb + tr_se - tr_esa)

   
        return response

    def calc_unrephasing(self, w3, w1, tau2):
        """Compute non-rephasing diagrams (+k1, -k2, +k3)."""
        if self._active_backend == "dense":
            return self._calc_unrephasing_dense(w3, w1, tau2)

        if not self._L_eff_sp:
            raise RuntimeError("Call feed_model() before calc_unrephasing().")

     
        response = np.zeros(self.N_k, dtype=np.complex128)

        for i_k in range(self.N_k):
            rho = self._rho_eq[i_k]

            v1 = self._JL_plus[i_k] @ rho
            v1 = self._solve_resolvent(i_k, w1, v1)

            mid_gsb = self._JL_minus[i_k] @ v1
            mid_gsb = self._propagate_tau2(i_k, tau2, mid_gsb)
            path_gsb = self._JL_plus[i_k] @ mid_gsb
            path_gsb = self._solve_resolvent(i_k, w3, path_gsb)

            mid_se_esa = self._JR_minus[i_k] @ v1
            mid_se_esa = self._propagate_tau2(i_k, tau2, mid_se_esa)

            path_se = self._JR_plus[i_k] @ mid_se_esa
            path_se = self._solve_resolvent(i_k, w3, path_se)

            path_esa = self._JL_plus[i_k] @ mid_se_esa
            path_esa = self._solve_resolvent(i_k, w3, path_esa)

            tr_gsb = self._trace_output(i_k, path_gsb)
            tr_se = self._trace_output(i_k, path_se)
            tr_esa = self._trace_output(i_k, path_esa)

            response[i_k] = -1j * (tr_gsb + tr_se - tr_esa)

     
        return response

    def generate_2D_spectra(
        self,
        w_list,
        tau2,
        k_array=None,
        n_jobs=None,
        parallel_backend=None,
        block_size=None,
        blas_threads=None,
        verbose=True,
    ):
        """
        Scan the w1/w3 grid and integrate over k.

        Parameters
        ----------
        parallel_backend : {"serial", "threading", "loky", "multiprocessing", "process"}
            Backend used for the omega_3 column loop. "process" uses a
            ProcessPoolExecutor initializer so the solver is copied once per
            worker instead of once per omega block.
        block_size : int or None
            Number of omega_3 columns per submitted task. Larger blocks reduce
            process-backend overhead.
        blas_threads : int or None
            Optional limit for nested BLAS/OpenMP threads during this scan.

        Returns
        -------
        dict
            Complex 2D response matrices for rephasing, non-rephasing, and
            absorptive spectra.
        """
        if n_jobs is None:
            n_jobs = self.n_jobs
        if self.N_k is None:
            raise RuntimeError("Call feed_model() before generate_2D_spectra().")

        
        self._resolvent_cache.clear()
        self._dense_resolvent_cache.clear()
        self._dense_tau2_cache.clear()

        w_list = np.asarray(w_list, dtype=float)
        n_w = len(w_list)

        if k_array is not None and self.N_k > 1:
            k_array = np.asarray(k_array, dtype=float)
            if len(k_array) != self.N_k:
                raise ValueError(
                    f"k_array has length {len(k_array)}, expected {self.N_k}"
                )
            dk = (k_array[-1] - k_array[0]) / len(k_array)
            integration_factor = dk / (2 * np.pi)
        else:
            integration_factor = 1.0

        S3_reph = np.zeros((n_w, n_w), dtype=np.complex128)
        S3_unreph = np.zeros((n_w, n_w), dtype=np.complex128)

        if parallel_backend is None:
            parallel_backend = self.parallel_backend
        n_jobs_eff = self._effective_n_jobs(n_jobs)
        blocks = self._make_w3_blocks(n_w, n_jobs_eff, block_size)

        if verbose:
            print(
                f"Starting 2D scan on a {n_w}x{n_w} frequency grid "
                f"with Liouville={self._active_backend}, "
                f"parallel={parallel_backend}, n_jobs={n_jobs_eff}, "
                f"block_size={len(blocks[0]) if blocks else 0}."
            )

        with self._parallel_context(blas_threads):
            results = self._run_w3_blocks(
                blocks,
                w_list,
                tau2,
                integration_factor,
                parallel_backend,
                n_jobs_eff,
            )
        

        for i, col_reph, col_unreph in results:
            S3_reph[:, i] = col_reph
            S3_unreph[:, i] = col_unreph
            
       
                
        return {
            "rephasing": S3_reph,
            "unrephasing": S3_unreph,
            "absorptive": S3_reph + S3_unreph,
        }

    def benchmark_2D_parallel_backends(
        self,
        w_list,
        tau2,
        k_array=None,
        backends=("serial", "threading", "loky", "process"),
        n_jobs_values=(1, 2, 4),
        block_size=None,
        blas_threads=1,
        repeats=1,
        max_w_points=None,
    ):
        """
        Benchmark omega-loop parallel backends for generate_2D_spectra().

        The first serial run is used as the numerical reference. The returned
        list contains elapsed time, speedup, and max absolute difference from
        that reference for each backend/job-count pair.
        """
        n_w = len(w_list)
        d = self.dim
        d2 = d**2 if d is not None else None

        print("\n--- 2D parallel benchmark context ---")
        print(f"Liouville backend      : {self._active_backend}")
        print(f"Hilbert dimension      : {d}")
        print(f"Liouville dimension    : {d2}")
        print(f"k-points               : {self.N_k}")
        print(f"omega points           : {n_w}")
        print(f"omega grid             : {n_w} x {n_w} = {n_w**2} omega pairs")
        print(f"tau2                   : {tau2}")
        print(f"Eta                    : {self.eta}")
        print(f"BLAS threads limit     : {blas_threads}")
        print(f"tested backends        : {backends}")
        print(f"tested n_jobs          : {n_jobs_values}")
        print(f"block_size             : {block_size}")
        print(f"repeats                : {repeats}")

        if self._active_backend == "sparse":
            print(f"sparse_solver          : {self.sparse_solver}")
            print(f"cache_resolvents       : {self.cache_resolvents}")
            print(f"cache_sparse_tau2      : {self.cache_sparse_tau2}")
            print(f"gmres rtol             : {self.sparse_gmres_rtol}")
            print(f"gmres maxiter          : {self.sparse_gmres_maxiter}")

        if self._active_backend == "dense":
            print(f"dense_liouville_cutoff : {self.dense_liouville_cutoff}")

        print("-------------------------------------\n")





        w_bench = np.asarray(w_list, dtype=float)
        if max_w_points is not None:
            w_bench = w_bench[: int(max_w_points)]

        reference = None
        reference_time = None
        rows = []

        for backend in backends:
            jobs_to_run = (1,) if backend == "serial" else n_jobs_values
            for n_jobs in jobs_to_run:
                best_elapsed = np.inf
                best_spectra = None

                for _ in range(max(1, int(repeats))):
                    t0 = time.perf_counter()
                    spectra = self.generate_2D_spectra(
                        w_bench,
                        tau2,
                        k_array=k_array,
                        n_jobs=n_jobs,
                        parallel_backend=backend,
                        block_size=block_size,
                        blas_threads=blas_threads,
                        verbose=False,
                    )
                    elapsed = time.perf_counter() - t0
                    if elapsed < best_elapsed:
                        best_elapsed = elapsed
                        best_spectra = spectra

                if reference is None:
                    reference = best_spectra
                    reference_time = best_elapsed

                max_abs_diff = float(
                    np.max(
                        np.abs(best_spectra["absorptive"] - reference["absorptive"])
                    )
                )
                speedup = reference_time / best_elapsed if best_elapsed > 0 else np.inf
                row = {
                    "backend": backend,
                    "n_jobs": self._effective_n_jobs(n_jobs),
                    "block_size": block_size,
                    "elapsed_s": best_elapsed,
                    "speedup_vs_serial": speedup,
                    "max_abs_diff": max_abs_diff,
                }
                rows.append(row)
                print(
                    f"{backend:>15} n_jobs={row['n_jobs']:<3} "
                    f"time={best_elapsed:8.3f}s "
                    f"speedup={speedup:6.2f}x "
                    f"max_abs_diff={max_abs_diff:.3e}"
                )

        return rows




class SpectroscopyPlotter:
    def __init__(self, w_list):
        self.w_list = w_list

    def plot_spectrum(
        self,
        S3_rephasing,
        S3_nonrephasing,
        levels=60,
        figsize=(12, 14),
        save_path=None,
    ):
        """
        Display a 3x2 figure with real, imaginary, and absolute spectra.
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        w = self.w_list

        self._plot_subplot(
            axes[0, 0],
            w,
            np.real(S3_rephasing),
            levels,
            "RdBu_r",
            r"Real / Rephasing",
            ylabel=r"$\omega_3$",
        )
        self._plot_subplot(
            axes[0, 1],
            w,
            np.real(S3_nonrephasing),
            levels,
            "RdBu_r",
            r"Real / Non-rephasing",
        )

        self._plot_subplot(
            axes[1, 0],
            w,
            np.imag(S3_rephasing),
            levels,
            "RdBu_r",
            r"Imaginary / Rephasing",
            ylabel=r"$\omega_3$",
        )
        self._plot_subplot(
            axes[1, 1],
            w,
            np.imag(S3_nonrephasing),
            levels,
            "RdBu_r",
            r"Imaginary / Non-rephasing",
        )

        self._plot_subplot(
            axes[2, 0],
            w,
            np.abs(S3_rephasing),
            levels,
            "magma",
            r"Absolute / Rephasing",
            xlabel=r"$\omega_1$",
            ylabel=r"$\omega_3$",
            vmin=0,
        )
        self._plot_subplot(
            axes[2, 1],
            w,
            np.abs(S3_nonrephasing),
            levels,
            "magma",
            r"Absolute / Non-rephasing",
            xlabel=r"$\omega_1$",
            vmin=0,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved at: {save_path}")

        plt.show()

    def _plot_subplot(self, ax, w, data, levels, cmap, title, xlabel=None, ylabel=None, vmin=None):
        """Plot one contour panel."""
        lim = np.max(np.abs(data))
        if vmin is None:
            vmin, vmax = -lim, lim
        else:
            vmax = lim

        contour = ax.contourf(w, w, data, levels, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        plt.colorbar(contour, ax=ax)

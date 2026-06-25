"""
Exemple complet d'utilisation de LiouvilleSpectroscopyV6 avec le modèle SOC.

Ce script reprend le modèle CuGeO3/SOC du notebook
``SOC_model_Liouville.ipynb`` et montre comment :

1. définir les paramètres physiques et numériques;
2. construire H(k), l'opérateur de courant J(k) et le dipôle mu(k);
3. définir les opérateurs de dissipation;
4. utiliser l'état thermique par défaut ou fournir rho(0);
5. demander à UFSS de générer les pathways R1-R6;
6. calculer les spectres rephasing, unrephasing et absorptif;
7. calculer et tracer les pathways individuels;
8. sélectionner la partie réelle, imaginaire ou le module.

Le script est volontairement explicite. Pour une utilisation de production,
les dictionnaires et fonctions de modèle peuvent être déplacés dans un module
séparé.

Important
---------
Le solver V6 traduit les diagrammes UFSS dans le formalisme fréquentiel
impulsif du solver. Par défaut, il accepte les diagrammes du troisième ordre
ayant l'ordre de pulses canonique (0, 1, 2). Le calcul des enveloppes de pulses
finies et de leur convolution n'est pas effectué par ce solver.
"""

import numpy as np
from LiouvilleSpectroscopyV6 import LiouvilleSpectroscopySolver,SpectroscopyPlotter



# =============================================================================
# 1. PARAMÈTRES DU MODÈLE PHYSIQUE
# =============================================================================

MODEL_PARAMS = {
    # -------------------------------------------------------------------------
    # Secteur orbital
    # -------------------------------------------------------------------------
    # Énergie de l'état orbital sombre par rapport à l'état fondamental.
    # Unité : la même unité d'énergie que toutes les autres énergies du modèle
    # (eV si le reste du modèle est exprimé en eV).
    "Delta_dark": 0.9,

    # Énergie de l'état orbital brillant.
    #
    # Attention : le notebook SOC utilisait parfois la clé "Delta_Bright".
    # La fonction build_soc_hamiltonian lit exactement "Delta_bright".
    "Delta_bright": 1.5,

    # Force du couplage spin-orbite effectif.
    "Lambda_SOC": 0.15,

    # Nombre de niveaux conservés dans le secteur bosonique/spin-Peierls.
    # La dimension totale de Hilbert est :
    #     3 états orbitaux x n_bosons.
    "n_bosons": 2,

    # -------------------------------------------------------------------------
    # Paramètres de la dispersion spin-Peierls
    # -------------------------------------------------------------------------
    # Température physique utilisée pour calculer le gap spin-Peierls.
    # Cette même valeur est aussi utilisée par le solver pour construire rho
    # thermique lorsque aucune matrice de densité initiale n'est fournie.
    "T": 7.0,

    # Température de transition spin-Peierls à champ nul.
    "T_SP_0": 14.0,

    # Champ magnétique externe.
    "B": 0.0,

    # Échelle d'échange du secteur spin.
    "J": 1.0,

    # Amplitude de dimérisation à température nulle.
    "delta_0": 0.01,

    # Exposant critique utilisé pour la dépendance en température de la
    # dimérisation.
    "beta": 0.5,

    # -------------------------------------------------------------------------
    # Couplage effectif associé à la dimérisation
    # -------------------------------------------------------------------------
    # Distance interne normalisée du dimère.
    "a_dimer": 1.0,

    # Amplitude de dimérisation utilisée dans le terme de couplage.
    "delta_dimerisation": 0.01,

    # Petit facteur de mélange/symmetry breaking.
    #
    # Attention : le notebook SOC utilisait parfois
    # "alpha_dimeraisation". La clé lue ici est "alpha_dimerisation".
    "alpha_dimerisation": 0.01,

    # Symétrie du couplage :
    #   "odd"   -> sin(k a/2)
    #   "even"  -> cos(k a/2)
    #   "mixed" -> sin(k a/2) + eta cos(k a/2)
    "parity": "odd",
}


# =============================================================================
# 2. PARAMÈTRES DE DISSIPATION
# =============================================================================

DISSIPATION_PARAMS = {
    # Taux de relaxation dans le secteur bosonique/spin.
    "gamma_spin": 0.1,

    # Taux de relaxation des états orbitaux sombre et brillant vers le
    # fondamental.
    "gamma_orb": 0.1,
}


# =============================================================================
# 3. PARAMÈTRES NUMÉRIQUES DU SOLVER
# =============================================================================

SOLVER_PARAMS = {
    # -------------------------------------------------------------------------
    # Résolution spectrale et état thermique
    # -------------------------------------------------------------------------
    # Élargissement imaginaire du résolvant :
    #     G(w) = [(w + i Eta) I - L]^{-1}.
    # Une valeur plus grande élargit et lisse les pics.
    "Eta": 0.1,

    # Température utilisée par l'état thermique par défaut.
    #
    # Si initial_density_matrix est fournie à feed_model(), cette température
    # n'est plus utilisée pour définir rho(0), mais elle peut encore être utile
    # au modèle physique puisque MODEL_PARAMS["T"] contrôle aussi la dispersion.
    "T": MODEL_PARAMS["T"],

    # Potentiel chimique utilisé uniquement lors de la construction de l'état
    # thermique par défaut.
    "mu": 0.0,

    # Tolérance utilisée pour séparer J_plus et J_minus dans l'approximation
    # des ondes tournantes.
    "rwa_tol": 1e-6,

    # Tolérance des validations de rho : hermiticité, trace et positivité.
    "density_matrix_tolerance": 1e-10,

    # -------------------------------------------------------------------------
    # Choix du backend de Liouville
    # -------------------------------------------------------------------------
    # "auto"   : dense pour les petits espaces de Liouville, sparse sinon.
    # "dense"  : force les tableaux denses.
    # "sparse" : force les matrices creuses.
    "backend": "auto",

    # En mode "auto", le backend dense est utilisé si dim(H)^2 est inférieur
    # ou égal à cette valeur.
    "dense_liouville_cutoff": 512,

    # -------------------------------------------------------------------------
    # Résolution sparse
    # -------------------------------------------------------------------------
    # "auto"   : essaie une factorisation directe puis GMRES si nécessaire.
    # "direct" : impose la factorisation directe.
    # "gmres"  : utilise le solveur itératif GMRES.
    "sparse_solver": "auto",

    # Tolérances et nombre maximal d'itérations de GMRES.
    "sparse_gmres_rtol": 1e-8,
    "sparse_gmres_atol": 0.0,
    "sparse_gmres_maxiter": None,

    # Si True, construit et conserve exp(-i L tau2) en sparse.
    # Cela accélère les appels répétés au même tau2, au prix de mémoire.
    "cache_sparse_tau2": False,

    # -------------------------------------------------------------------------
    # Cache des résolvants
    # -------------------------------------------------------------------------
    # Conserver les factorisations/résolvants associés aux fréquences.
    "cache_resolvents": True,

    # None conserve toutes les fréquences du scan courant. Un entier impose
    # une taille maximale au cache.
    "max_resolvent_cache": None,

    # -------------------------------------------------------------------------
    # Parallélisation de la grille (omega1, omega3)
    # -------------------------------------------------------------------------
    # Options principales :
    #   "serial"        : aucun parallélisme;
    #   "threading"     : threads joblib;
    #   "loky"          : processus joblib;
    #   "multiprocessing";
    #   "process"       : ProcessPoolExecutor.
    #
    # Pour un petit modèle dense, "threading" ou même "serial" peut être plus
    # efficace que les processus, qui imposent un coût de sérialisation.
    "parallel_backend": "threading",

    # Convention joblib :
    #   1  -> séquentiel;
    #  -1  -> tous les coeurs disponibles;
    #  -2  -> tous sauf un.
    "n_jobs": -1,

    # Nombre de colonnes omega3 regroupées dans une tâche parallèle.
    # None laisse le solver choisir automatiquement.
    "parallel_block_size": None,

    # Limite optionnelle du nombre de threads internes BLAS/OpenMP pendant le
    # parallélisme externe. Mettre 1 peut éviter la sur-souscription CPU.
    "blas_threads": 1,

    # -------------------------------------------------------------------------
    # Composantes calculées par generate_2D_spectra
    # -------------------------------------------------------------------------
    # "both"        : rephasing + unrephasing + absorptive;
    # "rephasing"   : rephasing uniquement;
    # "unrephasing" : unrephasing uniquement.
    "spectrum_components": "both",
}


# Dictionnaire pratique transmis aux fonctions qui lisent des paramètres de
# plusieurs catégories. Les clés de SOLVER_PARAMS écrasent les clés physiques
# identiques; ici T est volontairement identique dans les deux dictionnaires.
ALL_PARAMS = {
    **MODEL_PARAMS,
    **DISSIPATION_PARAMS,
    **SOLVER_PARAMS,
}


# =============================================================================
# 4. PARAMÈTRES DU SCAN
# =============================================================================

SCAN_PARAMS = {
    # Nombre de points de la zone de Brillouin.
    "total_kpoints": 100,

    # Bornes de k. Le solver intègre ensuite la réponse avec dk/(2 pi).
    "k_min": -np.pi,
    "k_max": np.pi,

    # Grille commune utilisée pour omega1 et omega3.
    "omega_min": 0.5,
    "omega_max": 2.5,
    "omega_points": 100,

    # Temps d'attente entre les deuxième et troisième interactions.
    "tau2": 3.0,
}


# =============================================================================
# 5. CONTRÔLES DE L'EXEMPLE
# =============================================================================

EXAMPLE_OPTIONS = {
    # "dipole" utilise mu_opt et interaction_type="dipole".
    # "current" utilise J_raw et interaction_type="current".
    "interaction_mode": "dipole",

    # Si False, le solver construit l'état thermique par défaut avec T et mu.
    # Si True, l'exemple fournit explicitement rho0_site.
    "use_custom_density_matrix": False,

    # Si True, UFSS génère les groupes rephasing et unrephasing standards.
    # Si False, V6 conserve ses six pathways standards par défaut.
    "generate_pathways_with_ufss": True,

    # Le calcul des six matrices individuelles ajoute du travail au calcul
    # principal. Désactiver si seuls les spectres totaux sont nécessaires.
    "calculate_individual_pathways": True,

    # Affichage des figures.
    "show_plots": True,
}


# =============================================================================
# 6. FONCTIONS DU MODÈLE SOC
# =============================================================================

def spin_peierls_dispersion(k, temperature, magnetic_field, exchange,
                            transition_temperature, delta_zero, beta):
    """
    Calculer la dispersion spin-Peierls epsilon(k) et le gap Delta.

    Parameters
    ----------
    k : float
        Vecteur d'onde.
    temperature : float
        Température physique T.
    magnetic_field : float
        Champ magnétique B.
    exchange : float
        Échelle d'échange J.
    transition_temperature : float
        Température de transition T_SP à champ nul.
    delta_zero : float
        Dimérisation à température nulle.
    beta : float
        Exposant critique de la dimérisation.
    """
    field_coefficient = 0.004
    t_sp = transition_temperature * (
        1.0 - field_coefficient * magnetic_field**2
    )
    t_sp = max(0.0, t_sp)

    if t_sp > 0.0 and temperature < t_sp:
        delta = delta_zero * (1.0 - temperature / t_sp) ** beta
    else:
        delta = 0.0

    # Relation de Cross-Fisher : Delta proportionnel à J delta^(2/3).
    gap = 2.0 * exchange * delta ** (2.0 / 3.0) if delta > 0 else 0.0
    velocity = np.pi * exchange / 2.0
    epsilon_k = np.sqrt(gap**2 + (velocity * np.sin(k)) ** 2)
    return epsilon_k, gap


def effective_soc_coupling(
    k,
    delta_dimerisation,
    alpha_dimerisation,
    a_dimer=1.0,
    parity="odd",
):
    """
    Calculer le facteur de forme du couplage SOC dans le réseau dimérisé.
    """
    eta = delta_dimerisation * alpha_dimerisation
    phase = k * a_dimer / 2.0

    if parity == "odd":
        return np.sin(phase)
    if parity == "even":
        return np.cos(phase)
    if parity == "mixed":
        return np.sin(phase) + eta * np.cos(phase)
    raise ValueError("parity doit être 'odd', 'even' ou 'mixed'.")


def build_soc_hamiltonian(k_array, params):
    """
    Construire H(k), J(k) et mu(k) pour le modèle SOC du notebook.

    Basis
    -----
    La base produit est organisée comme :

        |orbital> tensor |niveau bosonique>

    avec trois états orbitaux :

        0 : fondamental
        1 : sombre
        2 : brillant

    Returns
    -------
    H_raw : ndarray, shape (N_k, d, d)
        Hamiltonien dans la base site/produit.
    J_raw : ndarray, shape (N_k, d, d)
        Opérateur de courant dH/dk.
    mu_opt_array : ndarray, shape (N_k, d, d)
        Opérateur dipolaire reliant le fondamental et l'état brillant.
    """
    n_k = len(k_array)
    n_bosons = int(params.get("n_bosons", 2))
    dim = 3 * n_bosons

    lambda_soc = params.get("Lambda_SOC", 0.1)
    delta_dark = params.get("Delta_dark", 0.9)
    delta_bright = params.get("Delta_bright", 1.5)

    temperature = params.get("T", 5.0)
    magnetic_field = params.get("B", 0.0)
    exchange = params.get("J", 1.0)
    transition_temperature = params.get("T_SP_0", 14.0)
    delta_zero = params.get("delta_0", 0.1)
    beta = params.get("beta", 0.5)

    a_dimer = params.get("a_dimer", 1.0)
    delta_dimerisation = params.get("delta_dimerisation", 0.01)
    alpha_dimerisation = params.get("alpha_dimerisation", 0.01)
    parity = params.get("parity", "odd")

    H_raw = np.zeros((n_k, dim, dim), dtype=complex)
    J_raw = np.zeros((n_k, dim, dim), dtype=complex)
    mu_opt_array = np.zeros((n_k, dim, dim), dtype=complex)

    # Hamiltonien orbital local.
    H_orbital = np.zeros((3, 3), dtype=complex)
    H_orbital[1, 1] = delta_dark
    H_orbital[2, 2] = delta_bright

    # Mélange sombre-brillant induit par le SOC.
    L_operator = np.zeros((3, 3), dtype=complex)
    L_operator[1, 2] = 1.0
    L_operator[2, 1] = 1.0

    # Dipôle optique fondamental <-> brillant.
    mu_orbital = np.zeros((3, 3), dtype=complex)
    mu_orbital[0, 2] = 1.0
    mu_orbital[2, 0] = 1.0

    # Opérateur d'annihilation tronqué du secteur bosonique.
    annihilation = np.zeros((n_bosons, n_bosons), dtype=complex)
    for level in range(1, n_bosons):
        annihilation[level - 1, level] = np.sqrt(level)

    creation = annihilation.conj().T
    number_operator = creation @ annihilation
    displacement_operator = annihilation + creation

    identity_orbital = np.eye(3, dtype=complex)
    identity_spin = np.eye(n_bosons, dtype=complex)

    H_local = np.kron(H_orbital, identity_spin)
    mu_optical = np.kron(mu_orbital, identity_spin)

    velocity = np.pi * exchange / 2.0
    phase_derivative = a_dimer / 2.0
    eta_dimer = delta_dimerisation * alpha_dimerisation

    for i_k, k in enumerate(k_array):
        epsilon_k, _ = spin_peierls_dispersion(
            k,
            temperature,
            magnetic_field,
            exchange,
            transition_temperature,
            delta_zero,
            beta,
        )

        if epsilon_k != 0:
            d_epsilon_dk = (
                velocity**2 * np.sin(k) * np.cos(k) / epsilon_k
            )
        else:
            d_epsilon_dk = 0.0

        H_spin = np.kron(
            identity_orbital,
            epsilon_k * number_operator,
        )
        J_spin = np.kron(
            identity_orbital,
            d_epsilon_dk * number_operator,
        )

        coupling = effective_soc_coupling(
            k,
            delta_dimerisation,
            alpha_dimerisation,
            a_dimer,
            parity,
        )

        phase = k * a_dimer / 2.0
        if parity == "odd":
            d_coupling_dk = phase_derivative * np.cos(phase)
        elif parity == "even":
            d_coupling_dk = -phase_derivative * np.sin(phase)
        elif parity == "mixed":
            d_coupling_dk = (
                phase_derivative * np.cos(phase)
                - eta_dimer * phase_derivative * np.sin(phase)
            )
        else:
            raise ValueError("parity doit être 'odd', 'even' ou 'mixed'.")

        H_soc = (
            lambda_soc
            * coupling
            * np.kron(L_operator, displacement_operator)
        )
        J_soc = (
            lambda_soc
            * d_coupling_dk
            * np.kron(L_operator, displacement_operator)
        )

        H_raw[i_k] = H_local + H_spin + H_soc
        J_raw[i_k] = J_spin + J_soc
        mu_opt_array[i_k] = mu_optical

    return H_raw, J_raw, mu_opt_array


def build_minimal_lindblad_operators(params):
    """
    Construire les trois canaux de relaxation du notebook SOC.

    Returns
    -------
    list of tuple
        Chaque entrée est ``(C, gamma)`` :

        - ``C`` est un opérateur dans la base site/produit;
        - ``gamma`` est le taux Lindblad appliqué par le solver.

    Notes
    -----
    Les opérateurs retournés ne contiennent pas ``sqrt(gamma)``. Le solver
    reçoit gamma séparément et construit lui-même le dissipateur.
    """
    n_bosons = int(params.get("n_bosons", 2))
    gamma_spin = params.get("gamma_spin", 0.05)
    gamma_orbital = params.get("gamma_orb", 0.1)

    identity_orbital = np.eye(3, dtype=complex)
    identity_spin = np.eye(n_bosons, dtype=complex)

    annihilation = np.zeros((n_bosons, n_bosons), dtype=complex)
    for level in range(1, n_bosons):
        annihilation[level - 1, level] = np.sqrt(level)

    collapse_operators = []

    # Relaxation du secteur bosonique.
    C_spin = np.kron(identity_orbital, annihilation)
    collapse_operators.append((C_spin, gamma_spin))

    # Relaxation état brillant -> état fondamental.
    bright_to_ground = np.zeros((3, 3), dtype=complex)
    bright_to_ground[0, 2] = 1.0
    C_bright_ground = np.kron(bright_to_ground, identity_spin)
    collapse_operators.append((C_bright_ground, gamma_orbital))

    # Relaxation état sombre -> état fondamental.
    dark_to_ground = np.zeros((3, 3), dtype=complex)
    dark_to_ground[0, 1] = 1.0
    C_dark_ground = np.kron(dark_to_ground, identity_spin)
    collapse_operators.append((C_dark_ground, gamma_orbital))

    return collapse_operators


def build_custom_initial_density_matrix(params):
    """
    Exemple de rho(0) fourni dans la base site/produit.

    Ici, toute la population est placée dans :

        |orbital fondamental> tensor |niveau bosonique 0>.

    Pour définir un mélange statistique, placer plusieurs populations sur la
    diagonale. ``set_initial_density_matrix`` normalise la trace par défaut.
    Des cohérences peuvent être ajoutées hors diagonale si la matrice demeure
    hermitienne et positive semi-définie.
    """
    dimension = 3 * int(params.get("n_bosons", 2))
    rho0_site = np.zeros((dimension, dimension), dtype=complex)
    rho0_site[0, 0] = 1.0
    return rho0_site


# =============================================================================
# 7. PROGRAMME PRINCIPAL
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Grilles numériques
    # -------------------------------------------------------------------------
    k_array = np.linspace(
        SCAN_PARAMS["k_min"],
        SCAN_PARAMS["k_max"],
        SCAN_PARAMS["total_kpoints"],
    )
    w_list = np.linspace(
        SCAN_PARAMS["omega_min"],
        SCAN_PARAMS["omega_max"],
        SCAN_PARAMS["omega_points"],
    )
    tau2 = SCAN_PARAMS["tau2"]

    # -------------------------------------------------------------------------
    # Construction du modèle
    # -------------------------------------------------------------------------
    H_raw, J_raw, mu_opt = build_soc_hamiltonian(k_array, ALL_PARAMS)

    interaction_mode = EXAMPLE_OPTIONS["interaction_mode"].lower()
    if interaction_mode == "dipole":
        interaction_operator = mu_opt
        interaction_type = "dipole"
    elif interaction_mode == "current":
        interaction_operator = J_raw
        interaction_type = "current"
    else:
        raise ValueError(
            "interaction_mode doit être 'dipole' ou 'current'."
        )

    # -------------------------------------------------------------------------
    # Création du solver et choix de rho(0)
    # -------------------------------------------------------------------------
    system = LiouvilleSpectroscopySolver(SOLVER_PARAMS)

    if EXAMPLE_OPTIONS["use_custom_density_matrix"]:
        rho0_site = build_custom_initial_density_matrix(ALL_PARAMS)

        # rho0_site est exprimée dans la même base site/produit que H_raw.
        system.feed_model(
            H_raw,
            interaction_operator,
            interaction_type=interaction_type,
            initial_density_matrix=rho0_site,
            density_matrix_basis="site",
        )
    else:
        # En l'absence de rho explicite, V6 utilise l'état thermique construit
        # avec SOLVER_PARAMS["T"] et SOLVER_PARAMS["mu"].
        system.feed_model(
            H_raw,
            interaction_operator,
            interaction_type=interaction_type,
        )

    # Il est également possible de changer rho après feed_model :
    #
    # system.set_initial_density_matrix(rho0_site, basis="site")
    #
    # ou de restaurer l'état thermique :
    #
    # system.clear_initial_density_matrix()
    #
    # Inspection dans la base désirée :
    #
    # rho_eigen = system.get_initial_density_matrix(basis="eigen")
    # rho_site = system.get_initial_density_matrix(basis="site")

    # -------------------------------------------------------------------------
    # Dissipation
    # -------------------------------------------------------------------------
    collapse_operators = build_minimal_lindblad_operators(ALL_PARAMS)

    # Les opérateurs ont été construits dans la base site/produit.
    # Le solver les transforme dans la base propre de H(k).
    system.set_dissipation(collapse_operators, basis="site")

    # -------------------------------------------------------------------------
    # Pathways
    # -------------------------------------------------------------------------
    if EXAMPLE_OPTIONS["generate_pathways_with_ufss"]:
        # UFSS génère les deux conditions de phase standards :
        #
        # rephasing   : (-k1, +k2, +k3)
        # unrephasing : (+k1, -k2, +k3)
        #
        # Le quatrième temps représente la détection/local oscillator et
        # n'ajoute pas une interaction au pathway du troisième ordre.
        system.configure_standard_2d_pathways_with_ufss(
            arrival_times=[0.0, 100.0, 200.0, 300.0],
        )
    else:
        # Les six pathways standards intégrés à V6 restent actifs.
        system.reset_pathways()

    # Résumé pratique pour notebook/console. Chaque entrée contient :
    # name, component, interactions, pulse_indices, amplitude, bra_sign,
    # coefficient.
    pathway_definitions = system.pathway_summary()
    for pathway in pathway_definitions:
        print(pathway)

    # Exemple d'utilisation d'une liste produite ailleurs par UFSS :
    #
    # ufss_diagrams = diagram_generator.get_diagrams(arrival_times)
    # system.set_pathways_from_ufss(
    #     ufss_diagrams,
    #     component="rephasing",
    # )

    # -------------------------------------------------------------------------
    # Spectres totaux
    # -------------------------------------------------------------------------
    spectra = system.generate_2D_spectra(
        w_list,
        tau2=tau2,
        k_array=k_array,
        # Ces arguments remplacent temporairement les valeurs du dictionnaire
        # pour cet appel seulement.
        n_jobs=SOLVER_PARAMS["n_jobs"],
        parallel_backend=SOLVER_PARAMS["parallel_backend"],
        block_size=SOLVER_PARAMS["parallel_block_size"],
        blas_threads=SOLVER_PARAMS["blas_threads"],
        spectrum_components=SOLVER_PARAMS["spectrum_components"],
    )

    # Avec spectrum_components="both", le dictionnaire contient :
    rephasing_spectrum = spectra["rephasing"]
    unrephasing_spectrum = spectra["unrephasing"]
    absorptive_spectrum = spectra["absorptive"]

    # Les variables sont nommées ici pour montrer leur accès. Elles peuvent
    # ensuite être sauvegardées, analysées ou comparées.
    _ = (
        rephasing_spectrum,
        unrephasing_spectrum,
        absorptive_spectrum,
    )

    # -------------------------------------------------------------------------
    # Pathways individuels
    # -------------------------------------------------------------------------
    pathway_spectra = None
    if EXAMPLE_OPTIONS["calculate_individual_pathways"]:
        pathway_spectra = system.generate_2D_pathways(
            w_list,
            tau2=tau2,
            k_array=k_array,
            # Pour limiter le calcul à certains pathways :
            # pathways=["R1", "R2"],
        )

        # Accès direct à une matrice individuelle :
        R1_spectrum = pathway_spectra["R1"]
        _ = R1_spectrum

    # -------------------------------------------------------------------------
    # Tracés
    # -------------------------------------------------------------------------
    plotter = SpectroscopyPlotter(w_list)
    show = EXAMPLE_OPTIONS["show_plots"]

    # Un seul panneau, choix de la composante :
    plotter.plot_rephasing(
        spectra,
        component="real",
        show=show,
    )
    plotter.plot_unrephasing(
        spectra,
        component="imag",
        show=show,
    )
    plotter.plot_absorptive(
        spectra,
        component="abs",
        show=show,
    )

    # Alias disponible si la terminologie "nonrephasing" est préférée :
    #
    # plotter.plot_nonrephasing(spectra, component="real")

    if pathway_spectra is not None:
        # Tracer un seul pathway.
        plotter.plot_pathway(
            pathway_spectra,
            "R1",
            component="real",
            show=show,
        )

        # Tracer plusieurs pathways dans une grille.
        plotter.plot_pathways(
            pathway_spectra,
            pathways=["R1", "R2", "R3", "R4", "R5", "R6"],
            component="abs",
            ncols=3,
            show=show,
        )

    # Ancienne figure V5, toujours disponible :
    #
    # plotter.plot_spectrum(
    #     spectra["rephasing"],
    #     spectra["unrephasing"],
    # )


if __name__ == "__main__":
    main()

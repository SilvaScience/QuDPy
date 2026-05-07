import numpy as np 
import qutip as qt
import scipy.constants as const
import qudpy as qd
import ufss as uf

# 2. La spectroscopie 2D
from qudpy.Classes import System
import qudpy.plot_functions as pf

####################### Fonction qt #####################
"""
qt.tensor : Elle permet de combiner des opérateur agissant sur des sous-systèmes différents pour créer un opérateur global.
Exemple: Si l'espace A a une dimension N et l'espace B une dimension M, qt.tensor(A,B) crée une matrice de dimension (N X M)

Operateur de mode bosonique
Ces lignes définissent l'opérateur d'annihilation pour un mode bosonique

qt.destroy(N_fock): crée l'opérateur d'annihilation a dans un espace de fock tronqué à N_fock états.

qt.qeye(2) : C'est la matrice identité de taille 2. Elle sert ici de "placehold" pour dire: "ne fais rien sur le premier système (le spin)

Donc 
a= qt.tensor(qt.qeyes, qt.destroy(N_fock)) : L'opérateur a global (spin)X(operaeur boson)

a.dag(): methode calcule l'adjoint hermitien

Opérateur de spin à deux niveaux
Ici on défini les opérateur agissant sur le premier sous-système:

qt.destroy(2) :Pour un système de dimension 2, l'opérateur d'annihilation correspond à l'opérateur de descente sigma-

sm = qt.tensor (qt.destroy(2), qt.qeye(N_fock)) : Défini (simga-, Identié_boson). Cet opérateur fait baisser l'état du spin.

sp = sm.dag(): operateur sigma+

qt.sigmaz() : matrice de Pauli simga_z

sz = qt.tensor(qt.sigmaz(),qt.qete(N_fock)): L'opérateur simga_z étendu à tout l'espace de Hilbert


####################### Hamiltonien ########################################

H_elec : Opérateur sp*sm corespond a sigma_+, sigma_-, c'est l'opérateur de projection. Il vaut 1 si le système est dans l'état excité et 0
si le système est dans l'état fondamental. l'énergie de l'état excité est fixé à 1 (unité normalisé)

H_vib : C'est l'énergie de viration (l'oscillateur harmonique). 
- l'operateur a*adag est l'opérateur de nombre n. 
- w_vib est la fréquence de vibration. 
ce terme compte simplement combien de quanta de vibration (phonons) sont présents et leur attirbue une énergie homega chacun

H_couplage : C'est le terme d'interaction

sp*sm: Ce terme indique que le couplage n'existe que si l'électron est dans l'état excité. Si l'électron est au repos (0), le couplage s'annule

(a+adag): Cet opérateur est proportionnel à la position x de l'oscillateur

np.sqrt(S): Force du couplage

Résultat: Lorsque l'électron passe dans l'état excité, il exerce une force sur le réseau, ce qui déplace la position d'équilibre de la vibration. C'est ce qu'on appelle un oscillateur déplacé. 

L'Hamiltonien total décrit le scénario suivant: 
Au repos: le système voit un poteniel harmonique centré en x= 0

À l'excitation, : Le système gagne de l'énergie électronique, mais son poteniel harmonique est soudainement décalé dans l'espace à cause du terme de couplage

H = epsilon Ie><eI + h*omega*adag*a + h*omega*sqrt(S)* (a + adag)Ie><eI

"""

################### Constantes et paramètres du modèle ###################################

hbar = 0.658211951 # hbar en eV*fs
E_elec= 1.5 # Energie de la transition d-d
E_vib = 0.040 # Énegie du phonon 
S=2.5 # Facteur de Hunag-Rhys

w_e = E_elec / hbar #Freqeucne
w_v = E_vib / hbar

N_fock = 5 # Troncature des phonons/ nombre de quanta de vibration



#Operateur phononiques
a = qt.tensor(qt.qeye(2), qt.destroy(N_fock))
adag = a.dag()

# Opérateur électronique
sm = qt.tensor(qt.destroy(2), qt.qeye(N_fock))
sp = sm.dag()
sz = qt.tensor(qt.sigmaz(),qt.qeye(N_fock))

#operateur dipolaire
mu=1.0 * (sp + sm)

##################### Construction Hamiltonien ###########################################

H_elec = hbar*w_e * sp * sm 

H_vib = w_v *adag*a

# Le terme de couplage qui décale le potentiel pour l'état excité :
H_couplage = hbar*w_v * np.sqrt(S) * (a + adag) * (sp*sm)

H_total = H_elec + H_vib + H_couplage

######## Dissipation (Le bain) ###############################

kappa_elec = 0.02 # Taux de relaxation de l'électron 

kappa_vib = 0.05 # Taux de relaxation du phonon (déphasage vibratoire)

T = 20 # temperature en K
kB = 8.617333262e-5  # eV/K
beta = 1 / (T * kB)

# Population thermique du phonon à temperature choisi
n_th_vib= 1/(np.exp(E_vib * beta)-1)

# Matrice de dissipation (linblad)
c_ops = [
    #Relaxation électronique (vers le  fonamental pure)
    np.sqrt(kappa_elec)* sm,
    # Déphasage pur électronique (si nécessaire pour élargir)
    np.sqrt(kappa_elec *2)*sp*sm,
    # Relaxation du phonon (avec le bain thermique)
    np.sqrt(kappa_vib * n_th_vib + 1) * adag,
    #excitation themrique du phonon 
    np.sqrt(kappa_vib * n_th_vib) * adag
]

########### Matrice de densité #######################

rho_elec = qt.fock_dm(2,0)
rho_vib = qt.thermal_dm(N_fock,n_th_vib)
rho = qt.tensor(rho_elec,rho_vib)

# =====================================================================
# SÉQUENCE D'IMPULSIONS ET SPECTRES 2D
# =====================================================================

# 1. Initialisation du Système QuDPy
"""
On commence par créer le système avec la fonction de qudpy. Voici les paramètres nécessaire.
- H : Hamiltonien du système

- rho : etat initial du système (fondamental thermique)

- a , u : représente les perturbation ou pulse electrique. a représente l'absorption et u (up/transition) définissent comment le laser fait 
passer le sytème d'un autre niveau. 

- c_ops = Injection des opérateurs de Linblad, pour inclure la relaxation

-diagonalise = True : QuDpy calcule les valeur propres de H. Cela permet de passer dans la base probre, ce qui rend les calculs de propagation temporelle
infiniment plus rapides.
"""

sys = System(H=H_total, rho=rho, a=mu, u=mu, c_ops=c_ops, diagonalize=True)

# Paramétrage des délai temporelles (En femtosecondes)
# Pour résoudre un pohon de 103 fs, on échantillone environ sur 300fs

res_fs= 10 #résolution temporelle (pas de 3fs)
N_t1 = 100 # Nombre de pas durant le délai t1 (total= 300fs)
N_t3 = 100 # Nombre de pas durant la détection t3 (total = 300fs)
t2_attente = 0 #Temps de population (T) fixé à 0 fs pour commencer

time_delays = [N_t1,t2_attente, N_t3]
scan_id = [0,2] # On effectue la transformée de Fourier sur t1 (index 0 ) et t3 (index 2)

########## Calcul de propagation pour chaque diagramme ###################
######## Diagram ##############33

DG = uf.DiagramGenerator
R3rd = DG()
dummy_times = [0, 100, 200, 300]
t_pulse = np.array([-1, 1])
R3rd.efield_times = [t_pulse] * 4

# --- 1. Diagrammes Rephasing (R1, R2, R3) ---
# Phase discrimination pour k_I = -k1 + k2 + k3
R3rd.set_phase_discrimination([(0, 1), (1, 0), (1, 0)])
[R3, R1, R2] = R3rd.get_diagrams(dummy_times)
rephasing = [R1, R2, R3]

# --- 2. Diagrammes Non-Rephasing (R4, R5, R6) ---
# Phase discrimination pour k_II = +k1 - k2 + k3
R3rd.set_phase_discrimination([(1, 0), (0, 1), (1, 0)])
[R6, R4, R5] = R3rd.get_diagrams(dummy_times)
nonrephasing = [R4, R5, R6]
response_list = []
total_diagrams = rephasing + nonrephasing


print("Calcul de la dynamique quantique en cours...")
for k in range(6):
    # L'argument r=res_fs dicte le pas temporel à la librairie
    states, t1, t2, dipole = sys.coherence2d(time_delays, total_diagrams[k], scan_id, r=res_fs, parallel=True)
    response_list.append(1j * dipole)

######## Transformée de Fourier 2D pour passer au domaines fréquenceiel

spectra_list, extent, f1, f2 = sys.spectra(response_list, resolution=res_fs)

# Extraction et regroupement des signaux

rephasing_spectra = spectra_list[:3]
rephasing_spectra.append(np.sum(spectra_list[:3], 0)) # La somme donne le signal Rephasing macroscopique

nonrephasing_spectra = spectra_list[3:]
nonrephasing_spectra.append(np.sum(spectra_list[3:], 0))

print("Génération de la figure...")
# Affichage du signal Rephasing (le plus instructif pour l'élargissement inhomogène)


pf.silva_plot(rephasing_spectra, f1,f2, labels=None,
              title_list=['$R_1$', '$R_2$', '$R_3$', '$R_{rephasing}$'], scale='linear', color_map='PuOr',
              interpolation='spline36', center_scale=False, plot_sum=False, plot_quadrant='4', invert_y=False,
              diagonals=[True, True])
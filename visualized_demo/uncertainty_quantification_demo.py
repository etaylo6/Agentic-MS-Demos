import numpy as np
from constrainthg.hypergraph import Hypergraph

hg = Hypergraph()

P = hg.add_node('point load')
k = hg.add_node('kappa')
E = hg.add_node('youngs modulus')
I = hg.add_node('moment of inertia')
G = hg.add_node('shear modulus')
A = hg.add_node('area')
L = hg.add_node('length')
w = hg.add_node('deflection')
radius = hg.add_node('radius')
V = hg.add_node('poisson')
S = hg.add_node('slenderness ratio')
heuristic = hg.add_node('heuristic', static_value=10)

# E,V,G Relations
def shear_modulus_from_elastic_poisson(E, V):
    return E / (2 * (1 + V))

def elastic_modulus_from_shear_poisson(G, V):
    return 2 * G * (1 + V)

def poisson_from_elastic_shear(E, G):
    return (E / (2 * G)) - 1

# R,A Relations
def area_from_radius(radius):
    return np.pi * radius**2

def radius_from_area(A):
    return np.sqrt(A / np.pi)

# R,I Relations
def moment_of_inertia_from_radius(radius):
    return (np.pi / 4) * radius**4

def radius_from_moment_of_inertia(I):
    return (4 * I / np.pi)**(1/4)

def moment_of_inertia_from_area(A):
    return A**2 / (4 * np.pi)

def area_from_moment_of_inertia(I):
    return np.sqrt(4 * np.pi * I)

# Euler-Bernoulli Relations
def EB_deflection_for_T(P, L, E, I):
    return P * L**3 / (3 * E * I)

def EB_solve_for_P(w, L, E, I):
    return 3 * E * I * w / (L**3)

def EB_solve_for_E(w, P, L, I):
    return (P * L**3) / (3 * I * w)

def EB_solve_for_I(w, P, L, E):
    return (P * L**3) / (3 * E * w)

def EB_solve_for_L(w, P, E, I):
    return (3 * E * I * w / P) ** (1.0 / 3.0)

# Timoshenko Relations
def timoshenko_beam_deflection(P, E, I, L, k, G, A):
    delta_bending = P * L**3 / (3 * E * I)
    delta_shear = P * L / (k * G * A)
    return delta_bending + delta_shear

def solve_for_P_from_timoshenko(w, E, I, L, k, G, A):
    return w / (L**3/(3*E*I) + L/(k*G*A))

def solve_for_E_from_timoshenko(w, P, I, L, k, G, A):
    return P * L**3 / (3 * I * (w - P*L/(k*G*A)))

def solve_for_I_from_timoshenko(w, P, E, L, k, G, A):
    return P * L**3 / (3 * E * (w - P*L/(k*G*A)))

def solve_for_L_from_timoshenko(w, P, E, I, k, G, A):
    coeffs = [P/(3*E*I), 0, P/(k*G*A), -w]
    roots = np.roots(coeffs)
    real_positive_roots = roots[np.isreal(roots) & (roots > 0)]
    return real_positive_roots[0] if len(real_positive_roots) > 0 else 0

def solve_for_k_from_timoshenko(w, P, E, I, L, G, A):
    return P * L / (G * A * (w - P*L**3/(3*E*I)))

def solve_for_G_from_timoshenko(w, P, E, I, L, k, A):
    return P * L / (k * A * (w - P*L**3/(3*E*I)))

def solve_for_A_from_timoshenko(w, P, E, I, L, k, G):
    return P * L / (k * G * (w - P*L**3/(3*E*I)))
def slenderness_ratio(radius,L):
    return L / radius

# Add edges
hg.add_edge(sources=[E, V], target=G, rel=shear_modulus_from_elastic_poisson, label='E,V->G')
hg.add_edge(sources=[G, V], target=E, rel=elastic_modulus_from_shear_poisson, label='G,V->E')
hg.add_edge(sources=[E, G], target=V, rel=poisson_from_elastic_shear, label='E,G->V')

hg.add_edge(sources=[radius], target=A, rel=area_from_radius, label='radius->A')
hg.add_edge(sources=[A], target=radius, rel=radius_from_area, label='A->radius')

hg.add_edge(sources=[radius], target=I, rel=moment_of_inertia_from_radius, label='radius->I')
hg.add_edge(sources=[I], target=radius, rel=radius_from_moment_of_inertia, label='I->radius')

hg.add_edge(sources=[A], target=I, rel=moment_of_inertia_from_area, label='A->I')
hg.add_edge(sources=[I], target=A, rel=area_from_moment_of_inertia, label='I->A')

hg.add_edge(sources=[P, E, I, L, k, G, A], target=w, rel=timoshenko_beam_deflection, label='Timoshenko')
hg.add_edge(sources=[w, E, I, L, k, G, A], target=P, rel=solve_for_P_from_timoshenko, label='Timoshenko->P')
hg.add_edge(sources=[w, P, I, L, k, G, A], target=E, rel=solve_for_E_from_timoshenko, label='Timoshenko->E')
hg.add_edge(sources=[w, P, E, L, k, G, A], target=I, rel=solve_for_I_from_timoshenko, label='Timoshenko->I')
hg.add_edge(sources=[w, P, E, I, k, G, A], target=L, rel=solve_for_L_from_timoshenko, label='Timoshenko->L')
hg.add_edge(sources=[w, P, E, I, L, G, A], target=k, rel=solve_for_k_from_timoshenko, label='Timoshenko->k')
hg.add_edge(sources=[w, P, E, I, L, k, A], target=G, rel=solve_for_G_from_timoshenko, label='Timoshenko->G')
hg.add_edge(sources=[w, P, E, I, L, k, G], target=A, rel=solve_for_A_from_timoshenko, label='Timoshenko->A')

hg.add_edge(sources=[P, L, E, I, S, heuristic], target=w, rel=EB_deflection_for_T, label='EB: w = PL^3/(3EI)', via=lambda **kwargs: kwargs.get('slenderness_ratio', 0) > kwargs.get('heuristic', 0))
hg.add_edge(sources=[w, L, E, I, S, heuristic], target=P, rel=EB_solve_for_P, label='EB->P', via=lambda **kw: kw.get('slenderness_ratio', 0) > kw.get('heuristic', 0))
hg.add_edge(sources=[w, P, L, I, S, heuristic], target=E, rel=EB_solve_for_E, label='EB->E', via=lambda **kw: kw.get('slenderness_ratio', 0) > kw.get('heuristic', 0))
hg.add_edge(sources=[w, P, L, E, S, heuristic], target=I, rel=EB_solve_for_I, label='EB->I', via=lambda **kw: kw.get('slenderness_ratio', 0) > kw.get('heuristic', 0))
hg.add_edge(sources=[w, P, E, I, S, heuristic], target=L, rel=EB_solve_for_L, label='EB->L', via=lambda **kw: kw.get('slenderness_ratio', 0) > kw.get('heuristic', 0))

hg.add_edge(sources=[radius, L], target=S, rel=slenderness_ratio, label='ratio')

hg.solve(w, {'point load': 1000.0, 'youngs modulus': 200e9, 'length': 1.0, 'kappa': 5/6, 'area': 1e-4}, to_print=True)

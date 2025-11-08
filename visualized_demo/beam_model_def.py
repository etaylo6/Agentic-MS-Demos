import numpy as np
from constrainthg.hypergraph import Hypergraph, Node

import edge_grouper

DEFAULT_EDGE_PREFIXES = {
    "[MAT]": "material_properties",
    "[GEO-A]": "geometry_area",
    "[GEO-I]": "geometry_moi",
    "[TIM]": "beam_timoshenko",
    "[EB]": "beam_euler_bernoulli",
    "[SLEN]": "geometry_slenderness",
}

edge_grouper.set_default_prefix_map(DEFAULT_EDGE_PREFIXES)

def create_beam_model():
    hg = Hypergraph()

    P = Node('P')
    k = Node('k')
    E = Node('E')
    I = Node('I')
    G = Node('G')
    A = Node('A')
    L = Node('L')
    w = Node('w')
    radius = Node('radius')
    V = Node('V')
    S = Node('S')
    heuristic = Node('heuristic', static_value=10)
    
    # E,V,G Relations
    def modulus_G(E, V):
        return E / (2 * (1 + V))
    def modulus_E(G, V):
        return 2 * G * (1 + V)
    def modulus_V(E, G):
        return (E / (2 * G)) - 1
    
    # R,A Relations
    def Area_A(radius):
        return np.pi * radius**2
    def Area_radius(A):
        return np.sqrt(A / np.pi)
    
    # R,I Relations
    def MOI_I(radius):
        return (np.pi / 4) * radius**4
    def MOI_radius(I):
        return (4 * I / np.pi)**(1/4)
    def MOI_A_I(A):
        return A**2 / (4 * np.pi)
    def MOI_A_A(I):
        return np.sqrt(4 * np.pi * I)
    
    # Euler-Bernoulli Relations
    def EB_w(s1, s2, s3, s4, s5, s6):
        return s1 * s2**3 / (3 * s3 * s4)
    def EB_P(s1, s2, s3, s4, s5, s6):
        return 3 * s3 * s4 * s1 / (s2**3)
    def EB_E(s1, s2, s3, s4, s5, s6):
        return (s2 * s3**3) / (3 * s4 * s1)
    def EB_I(s1, s2, s3, s4, s5, s6):
        return (s2 * s3**3) / (3 * s4 * s1)
    def EB_L(s1, s2, s3, s4, s5, s6):
        return (3 * s3 * s4 * s1 / s2) ** (1.0 / 3.0)
    
    # Timoshenko Relations
    def timoshenko_w(P, E, I, L, k, G, A):
        delta_bending = P * L**3 / (3 * E * I)
        delta_shear = P * L / (k * G * A)
        return delta_bending + delta_shear
    def timoshenko_P(w, E, I, L, k, G, A):
        return w / (L**3/(3*E*I) + L/(k*G*A))
    def timoshenko_E(w, P, I, L, k, G, A):
        return P * L**3 / (3 * I * (w - P*L/(k*G*A)))
    def timoshenko_I(w, P, E, L, k, G, A):
        return P * L**3 / (3 * E * (w - P*L/(k*G*A)))
    def timoshenko_L(w, P, E, I, k, G, A):
        coeffs = [P/(3*E*I), 0, P/(k*G*A), -w]
        roots = np.roots(coeffs)
        real_positive_roots = roots[np.isreal(roots) & (roots > 0)]
        return real_positive_roots[0] if len(real_positive_roots) > 0 else 0
    def timoshenko_K(w, P, E, I, L, G, A):
        return P * L / (G * A * (w - P*L**3/(3*E*I)))
    def timoshenko_G(w, P, E, I, L, k, A):
        return P * L / (k * A * (w - P*L**3/(3*E*I)))
    def timoshenko_A(w, P, E, I, L, k, G):
        return P * L / (k * G * (w - P*L**3/(3*E*I)))
    
    # Slenderness Ratio
    def slenderness_ratio(**kwargs):
        L_val = kwargs.get('L')
        radius_val = kwargs.get('radius')
        return L_val / radius_val

    # Add edges normally with tagged labels
    # Material Properties: E, V, G relationships
    hg.add_edge([E, V], G, rel=modulus_G, label='[MAT] E,V->G')
    hg.add_edge([G, V], E, rel=modulus_E, label='[MAT] G,V->E')
    hg.add_edge([E, G], V, rel=modulus_V, label='[MAT] E,G->V')

    # Geometry: Area relationships
    hg.add_edge([radius], A, rel=Area_A, label='[GEO-A] radius->A')
    hg.add_edge([A], radius, rel=Area_radius, label='[GEO-A] A->radius')

    # Geometry: Moment of Inertia relationships
    hg.add_edge([radius], I, rel=MOI_I, label='[GEO-I] radius->I')
    hg.add_edge([I], radius, rel=MOI_radius, label='[GEO-I] I->radius')
    hg.add_edge([A], I, rel=MOI_A_I, label='[GEO-I] A->I')
    hg.add_edge([I], A, rel=MOI_A_A, label='[GEO-I] I->A')

    # Timoshenko Beam Theory
    hg.add_edge([P, E, I, L, k, G, A], w, rel=timoshenko_w, label='[TIM] w')
    hg.add_edge([w, E, I, L, k, G, A], P, rel=timoshenko_P, label='[TIM] P')
    hg.add_edge([w, P, I, L, k, G, A], E, rel=timoshenko_E, label='[TIM] E')
    hg.add_edge([w, P, E, L, k, G, A], I, rel=timoshenko_I, label='[TIM] I')
    hg.add_edge([w, P, E, I, k, G, A], L, rel=timoshenko_L, label='[TIM] L')
    hg.add_edge([w, P, E, I, L, G, A], k, rel=timoshenko_K, label='[TIM] k')
    hg.add_edge([w, P, E, I, L, k, A], G, rel=timoshenko_G, label='[TIM] G')
    hg.add_edge([w, P, E, I, L, k, G], A, rel=timoshenko_A, label='[TIM] A')

    # Euler-Bernoulli Beam Theory (slender beams: S >= heuristic)
    hg.add_edge({'s1': P, 's2': L, 's3': E, 's4': I, 's5': S, 's6': heuristic}, w, rel=EB_w, label='[EB] w', via=lambda s1, s2, s3, s4, s5, s6, **kwargs: s5 >= s6)
    hg.add_edge({'s1': w, 's2': L, 's3': E, 's4': I, 's5': S, 's6': heuristic}, P, rel=EB_P, label='[EB] P', via=lambda s1, s2, s3, s4, s5, s6, **kwargs: s5 >= s6)
    hg.add_edge({'s1': w, 's2': P, 's3': L, 's4': I, 's5': S, 's6': heuristic}, E, rel=EB_E, label='[EB] E', via=lambda s1, s2, s3, s4, s5, s6, **kwargs: s5 >= s6)
    hg.add_edge({'s1': w, 's2': P, 's3': L, 's4': E, 's5': S, 's6': heuristic}, I, rel=EB_I, label='[EB] I', via=lambda s1, s2, s3, s4, s5, s6, **kwargs: s5 >= s6)
    hg.add_edge({'s1': w, 's2': P, 's3': E, 's4': I, 's5': S, 's6': heuristic}, L, rel=EB_L, label='[EB] L', via=lambda s1, s2, s3, s4, s5, s6, **kwargs: s5 >= s6)

    # Slenderness Ratio
    hg.add_edge({'L': L, 'radius': radius}, S, rel=slenderness_ratio, label='[SLEN] S')
    
    # Return hypergraph and nodes
    nodes = {
        'P': P, 'k': k, 'E': E, 'I': I, 'G': G, 
        'A': A, 'L': L, 'w': w, 'radius': radius, 
        'V': V, 'S': S, 'heuristic': heuristic
    }
    
    return hg, nodes


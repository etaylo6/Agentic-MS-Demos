"""
Test function to verify Timoshenko beam calculations.
This will help identify calculation issues by testing with known values.
"""

import numpy as np
from variant_beam_model import create_timoshenko_hypergraph

def test_timoshenko_calculations():
    """
    Test the Timoshenko beam calculations with known values.
    """
    print("=" * 60)
    print("TESTING TIMOSHENKO BEAM CALCULATIONS")
    print("=" * 60)
    
    # Create hypergraph
    hg = create_timoshenko_hypergraph()
    
    # Test values (realistic engineering values)
    test_inputs = {
        "point load": 1000.0,           # 1 kN load
        "length": 2.0,                  # 2 m beam length
        "youngs modulus": 200000000000.0,  # 200 GPa (steel)
        "moment of inertia": 0.0001,    # 0.0001 m^4
        "shear modulus": 80000000000.0, # 80 GPa (steel)
        "area": 0.01,                   # 0.01 m^2
        "kappa": 5/6,                   # 0.833... (rectangular section)
        "poisson": 0.3                  # Poisson's ratio for steel
    }
    
    print("Test Input Values:")
    for key, value in test_inputs.items():
        print(f"  {key}: {value}")
    print()
    
    # Test 1: Direct calculation of shear modulus from E and V
    print("Test 1: Shear Modulus Calculation")
    E = test_inputs["youngs modulus"]
    V = test_inputs["poisson"]
    G_calculated = E / (2 * (1 + V))
    G_expected = test_inputs["shear modulus"]
    
    print(f"  E = {E}")
    print(f"  V = {V}")
    print(f"  G_calculated = E/(2*(1+V)) = {G_calculated}")
    print(f"  G_expected = {G_expected}")
    print(f"  Difference = {abs(G_calculated - G_expected)}")
    print(f"  Match: {abs(G_calculated - G_expected) < 1e6}")
    print()
    
    # Test 2: Direct Timoshenko beam deflection calculation
    print("Test 2: Direct Timoshenko Beam Deflection")
    P = test_inputs["point load"]
    E = test_inputs["youngs modulus"]
    I = test_inputs["moment of inertia"]
    L = test_inputs["length"]
    k = test_inputs["kappa"]
    G = test_inputs["shear modulus"]
    A = test_inputs["area"]
    
    # Calculate deflection components
    delta_bending = P * L**3 / (3 * E * I)
    delta_shear = P * L / (k * G * A)
    delta_total = delta_bending + delta_shear
    
    print(f"  P = {P} N")
    print(f"  L = {L} m")
    print(f"  E = {E} Pa")
    print(f"  I = {I} m^4")
    print(f"  k = {k}")
    print(f"  G = {G} Pa")
    print(f"  A = {A} m^2")
    print()
    print(f"  delta_bending = P*L^3/(3*E*I) = {delta_bending} m")
    print(f"  delta_shear = P*L/(k*G*A) = {delta_shear} m")
    print(f"  delta_total = {delta_total} m")
    print(f"  delta_total = {delta_total * 1000} mm")
    print()
    
    # Test 3: Hypergraph solve
    print("Test 3: Hypergraph Solve")
    try:
        # Convert to string-keyed dictionary for hypergraph
        inputs_dict = {str(k): v for k, v in test_inputs.items()}
        
        # Solve for theta (deflection)
        result = hg.solve("theta", inputs_dict, search_depth=1000)
        
        if result is not None:
            print(f"  Hypergraph solve successful!")
            print(f"  Result: {result}")
            
            # Get computed results
            if hasattr(hg, 'solved_tnodes') and hg.solved_tnodes:
                computed_results = {}
                for tnode in hg.solved_tnodes:
                    if hasattr(tnode, 'value') and tnode.value is not None:
                        computed_results[tnode.node_label] = tnode.value
                
                print(f"  Computed results: {computed_results}")
                
                if "theta" in computed_results:
                    hg_theta = computed_results["theta"]
                    print(f"  Hypergraph theta: {hg_theta}")
                    print(f"  Direct calculation theta: {delta_total}")
                    print(f"  Difference: {abs(hg_theta - delta_total)}")
                    print(f"  Match: {abs(hg_theta - delta_total) < 1e-6}")
        else:
            print("  Hypergraph solve failed!")
            
    except Exception as e:
        print(f"  Error in hypergraph solve: {e}")
    
    # Test 4: Check parameter mapping
    print("\nTest 4: Parameter Mapping Check")
    print("Expected parameter order: P, E, I, L, k, G, A")
    print("Expected values:")
    print(f"  P = {test_inputs['point load']}")
    print(f"  E = {test_inputs['youngs modulus']}")
    print(f"  I = {test_inputs['moment of inertia']}")
    print(f"  L = {test_inputs['length']}")
    print(f"  k = {test_inputs['kappa']}")
    print(f"  G = {test_inputs['shear modulus']}")
    print(f"  A = {test_inputs['area']}")
    print("\nFrom hypergraph debug output, we saw:")
    print("  P=1000.0, E=200000000000.0, I=0.01, L=0.833, k=2.0, G=80000000000.0, A=0.0001")
    print("This suggests the parameters are being swapped!")
    
    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

def test_with_different_values():
    """
    Test with different sets of values to identify issues.
    """
    print("\n" + "=" * 60)
    print("TESTING WITH DIFFERENT VALUE SETS")
    print("=" * 60)
    
    # Test set 1: Simple values
    print("Test Set 1: Simple Values")
    simple_inputs = {
        "point load": 100.0,
        "length": 1.0,
        "youngs modulus": 1000000000.0,  # 1 GPa
        "moment of inertia": 0.001,
        "shear modulus": 400000000.0,    # 0.4 GPa
        "area": 0.1,
        "kappa": 0.8,
        "poisson": 0.25
    }
    
    P, L, E, I, k, G, A = [simple_inputs[key] for key in 
                          ["point load", "length", "youngs modulus", "moment of inertia", 
                           "kappa", "shear modulus", "area"]]
    
    delta_bending = P * L**3 / (3 * E * I)
    delta_shear = P * L / (k * G * A)
    delta_total = delta_bending + delta_shear
    
    print(f"  Simple values result: {delta_total} m = {delta_total * 1000} mm")
    print()
    
    # Test set 2: Check if the issue is with the specific values
    print("Test Set 2: Check G calculation")
    E = 200000000000.0
    V = 0.3
    G = E / (2 * (1 + V))
    print(f"  E = {E}")
    print(f"  V = {V}")
    print(f"  G = E/(2*(1+V)) = {G}")
    print(f"  G in GPa = {G / 1e9}")
    print()

if __name__ == "__main__":
    test_timoshenko_calculations()
    test_with_different_values()

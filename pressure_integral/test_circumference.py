"""
test_circumference.py

Tests poloidal_circum on the circular case: eps=0.5, kap=1, dlt=0.
The zero contour is a circle of radius eps=0.5 centred at (1,0),
so the exact circumference is 2*pi*eps = pi.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pressure_utils import make_psi, poloidal_circum

EPS, KAP, DLT = 0.5, 1.0, 0.0
EXACT = 2 * np.pi * EPS          # pi

psi, _, _ = make_psi(EPS, KAP, DLT)

print(f"Circular case: eps={EPS}, kap={KAP}, dlt={DLT}")
print(f"Exact circumference = 2*pi*eps = {EXACT:.10f}\n")

print(f"{'N':>6}  {'circumference':>16}  {'error':>12}  {'rel. error':>12}")
print('-' * 52)

for N in [100, 200, 500, 1000, 2000]:
    C = poloidal_circum(psi, x_lim=(0.4, 1.6), y_lim=(-0.6, 0.6), N=N)
    err = abs(C - EXACT)
    print(f"{N:>6}  {C:>16.10f}  {err:>12.2e}  {err/EXACT:>12.2e}")

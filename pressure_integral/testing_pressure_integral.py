"""
testing_pressure_integral.py

Runs test_method_3 (int_contour_boundary) and prints a comparison table.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
import csv
import numpy as np

warnings.filterwarnings('ignore')   # suppress log(0) RuntimeWarnings

from pressure_integral.test_method_3 import collect_results as m3

results = m3()

def fmt_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    return f'{v:.10f}'

def fmt_err(v, a):
    if a is None or v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    return f'{abs(v - a):.2e}'

out_path = os.path.join(os.path.dirname(__file__), 'results.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Case', 'Method', 'Resolution', 'Numerical', 'Analytical', 'Error', 'Conv. Order'])
    for r in results:
        writer.writerow([
            r['case'], r['method'], r['res'],
            fmt_num(r['value']),
            fmt_num(r['analytical']),
            fmt_err(r['value'], r['analytical']),
            r['conv'],
        ])

print(f'Saved → {out_path}')

"""
Backward-compatible Colab import path.

Prefer:  from real_fractal_sim import run_colab
This module re-exports the same API for existing notebooks.
"""

from real_fractal_sim import *  # noqa: F401,F403
from real_fractal_sim import run_colab, print_report  # noqa: F401
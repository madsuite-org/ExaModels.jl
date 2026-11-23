"""Evaluate a compiled ExaModelsCompiler library through the Python consumer.

Loads the shared library with `cnlplib` (https://github.com/madsuite-org/cnlplib-py)
and writes its readings to a text file for the Julia side to compare against
`ExaModels` directly. Kept deliberately dependency-light — plain text out, no
JSON — so the test needs nothing beyond numpy.

    python3 cnlplib_check.py <libpath> <prefix> <n> <outfile>
"""

import sys
import numpy as np

try:
    import cnlplib
except ImportError:   # before the rename to cnlplib
    import cnlpmodels as cnlplib


def main():
    libpath, prefix, n, outfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

    lib = cnlplib.load(libpath)
    m = cnlplib.CModel(lib, n, prefix=prefix)

    # The same point the Julia side uses, generated identically rather than
    # passed across, so a mismatch in the point itself would show up as a
    # mismatch in the values.
    x = np.linspace(0.5, 3.0, m.nvar)
    y = np.linspace(-1.0, 1.0, m.ncon) if m.ncon else np.zeros(0)

    jr, jc = m.jac_structure()
    hr, hc = m.hess_structure()

    def row(name, v):
        v = np.atleast_1d(np.asarray(v)).ravel()
        return "%s %s" % (name, " ".join(repr(float(t)) for t in v))

    lines = [
        row("nvar", m.nvar), row("ncon", m.ncon),
        row("nnzj", m.nnzj), row("nnzh", m.nnzh),
        row("x0", m.x0), row("lvar", m.lvar), row("uvar", m.uvar),
        row("lcon", m.lcon), row("ucon", m.ucon),
        row("obj", m.obj(x)), row("grad", m.grad(x)),
        row("cons", m.cons(x) if m.ncon else np.zeros(0)),
        # Shift the 0-based Python convention back to the 1-based one the ABI
        # and ExaModels use, so the two sides are comparing the same thing.
        row("jac_rows", jr + 1), row("jac_cols", jc + 1), row("jac", m.jac(x)),
        row("hess_rows", hr + 1), row("hess_cols", hc + 1),
        row("hess", m.hess(x, y, obj_weight=0.5)),
    ]
    with open(outfile, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("ok %d %d" % (m.nvar, m.ncon))


if __name__ == "__main__":
    main()

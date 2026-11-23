# Drives the schema + builder ABI of a compiled structured model from Python:
# positional values against the schema's field order, the table as a dict of
# columns. Inputs mirror the S_* constants in runtests.jl — keep them in sync.
#
# usage: python builder_check.py <libpath> <prefix> <n> <outfile>
import sys

import numpy as np

try:
    import cnlplib
except ImportError:   # before the rename to cnlplib
    import cnlpmodels as cnlplib

libpath, prefix, n, outfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

lib = cnlplib.load(libpath)
v0 = np.linspace(0.1, 0.6, n)
lo = np.full(n, -5.0)
tab = {
    "i": np.array([2, 5, 6]),
    "w": np.array([1.5, 3.0, 0.5]),
    "s": np.array([2.0, -1.0, 0.0]),
}

m = cnlplib.CModel(lib, n, v0, lo, tab, prefix=prefix)
x = np.linspace(0.5, 3.0, n)

with open(outfile, "w") as f:
    f.write("nvar %d\n" % m.nvar)
    f.write("ncon %d\n" % m.ncon)
    f.write("obj %.17g\n" % m.obj(x))
    f.write("grad " + " ".join("%.17g" % v for v in m.grad(x)) + "\n")
    f.write("cons " + " ".join("%.17g" % v for v in m.cons(x)) + "\n")

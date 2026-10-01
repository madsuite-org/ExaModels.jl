Standalone correctness checks for the compile-time prototypes on this branch.
Not wired into runtests.jl. Run each in an environment with this ExaModels
checkout dev'ed plus NLPModels:

    julia --project=<env> refstest.jl    # non-concrete refs storage
    julia --project=<env> mergetest.jl   # same-type block merge vs single-block ground truth
    julia --project=<env> lifttest.jl    # add_expr(...; lift = true) vs spliced reference

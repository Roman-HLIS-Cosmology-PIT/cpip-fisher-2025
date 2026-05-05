import os, sys
from cocoa_dvcalc import SPACE, Provider, CoordDescPCA

space, i_mv, n_iter = sys.argv[1:4]
assert space == SPACE, f"Space mismatch. Expected: {SPACE}, got: {space}."
i_mv = int(i_mv)  # Index of the model vector.
n_iter = int(n_iter)  # Number of iteration or meta-iterations.

Provider.INPUT_FILE = os.path.join(
    os.path.dirname(Provider.INPUT_FILE),
    f"EXAMPLE_EVALUATE_v{i_mv}.yaml".replace("_v1.", "."))
filename= f"./projects/cocoa_dvcalc/dc1_{space}_v{i_mv}_map.json"  # _map

shrink_init = 0
coorddesc = CoordDescPCA(filename, shrink_init=shrink_init)
print(coorddesc.current_logp)
coorddesc.driver(n_meta=1, pca_mode=False, maxiter=10)

while True:
    for rescale in [True, False]:
        CoordDescPCA.RESCALE = rescale
        coorddesc = CoordDescPCA(filename, shrink_init=shrink_init)
        print(coorddesc.current_logp)
        coorddesc.driver(n_meta=10, pca_mode=True, maxiter=1)

    if coorddesc.shrink_count == shrink_init + 10:
        shrink_init += 1
        if shrink_init > 20: break

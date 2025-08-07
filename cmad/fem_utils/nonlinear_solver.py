from cmad.fem_utils.fem_problem import fem_problem
from cmad.fem_utils.neo_hookean import Neo_hookean
from cmad.fem_utils.thermoelastic import Thermoelastic
from cmad.fem_utils.thermo import Thermo
import numpy as np
import scipy.sparse.linalg
import scipy.sparse as sp

order = 2
problem = fem_problem("hole_block_traction", order, mixed=True)
num_steps, dt = problem.num_steps()

max_iters = 4
tol = 2e-10

model = Neo_hookean(problem)
# model.initialize_variables()

for step in range(num_steps):
    print('Step: ', step)
    model.compute_surf_tractions(step)
    model.set_prescribed_dofs(step)
    for i in range(max_iters):

        model.set_global_fields()

        model.seed_none()
        model.evaluate()
        RF = model.scatter_rhs()

        print("||R||: ", np.linalg.norm(RF))
        if (np.linalg.norm(RF) < tol):
            break

        model.seed_U()
        model.evaluate()
        KFF = model.scatter_lhs()

        # diag = KFF.diagonal()
        # T = sp.diags(1 / np.sqrt(diag), 0)

        # delta = scipy.sparse.linalg.spsolve(T @ KFF @ T, -T.dot(RF))
        delta = scipy.sparse.linalg.spsolve(KFF, -RF)

        model.add_to_UF(delta)

    model.save_global_fields()
    # model.advance_model()

# point_data = model.get_data()
# problem.save_data("rect_prism_thermoelastic_1.xdmf", point_data)













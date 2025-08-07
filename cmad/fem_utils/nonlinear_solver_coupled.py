from cmad.fem_utils.fem_problem import fem_problem
from cmad.fem_utils.elastic_plastic_small import Elastic_plastic_small
from cmad.fem_utils.elastic_plastic_small_plane_stress import Elastic_plastic_small_plane_stress
import numpy as np
import scipy.sparse.linalg
import time

def newton_solve(model, num_steps, max_iters, tol):
    model.initialize_plot()
    for step in range(num_steps):
        print('Step: ', step)
        model.set_prescribed_dofs(step)
        model.compute_surf_tractions(step)
        for i in range(max_iters):
            model.set_global_fields()

            model.compute_local_state_variables()
            model.evaluate_local()
            model.get_num_plastic_elements()

            model.evaluate_global()
            RF = model.scatter_rhs()
            print('Iteration: ', i)
            print(" ||C|| = ", np.max(np.abs(model.C())),
                  " ||R|| = ", np.linalg.norm(RF))
            if (np.linalg.norm(RF) < tol):
                break

            model.evaluate_tang()
            KFF = model.scatter_lhs()

            delta = scipy.sparse.linalg.spsolve(KFF, -RF)

            model.add_to_UF(delta)
            model.update_plot()

            model.reset_xi()

        model.save_global_fields()
        model.advance_model()
        model.update_plot()

def newton_solve_line_search(model, num_steps, max_iters, tol, s=0.8, m=8):
    model.initialize_plot()
    for step in range(num_steps):
        print('Timestep', step)
        # set displacement BCs
        model.set_prescribed_dofs(step)
        model.set_global_fields()

        #set traction BCs
        model.compute_surf_tractions(step)

        model.reset_xi()
        print("Computing local state variables...")
        model.compute_local_state_variables()
        model.evaluate_local()
        print("||C|| = ", np.max(np.abs(model.C())))

        model.evaluate_global()
        RF = model.scatter_rhs()

        for i in range(max_iters):
            print('Newton Iteration: ', i)
            print("||R|| = ", np.linalg.norm(RF))
            if (np.linalg.norm(RF) < tol):
                break

            model.evaluate_tang()
            KFF = model.scatter_lhs()
            KFF_factorized = scipy.sparse.linalg.factorized(KFF)
            delta = KFF_factorized(-RF)

            UF_curr = model.get_UF()
            UF_new = UF_curr + delta
            model.set_UF(UF_new)
            model.set_global_fields()

            model.reset_xi()
            print("Computing local state variables...")
            model.compute_local_state_variables()
            model.evaluate_local()
            print("||C|| = ", np.max(np.abs(model.C())))

            model.evaluate_global()
            RF_new = model.scatter_rhs()
            if (np.abs(np.dot(delta, RF_new)) <= s * np.abs(np.dot(delta, RF))):
                print('No line search')
                RF = RF_new.copy()
            else:
                print('Performing line search')
                for j in range(1, m + 1):
                    print("Line search iteration: ", j)
                    eta = (m - j + 1) / (m + 1)
                    UF_new = UF_curr + eta * delta
                    model.set_UF(UF_new)
                    model.set_global_fields()

                    model.reset_xi()
                    print("Computing local state variables...")
                    model.compute_local_state_variables()
                    model.evaluate_local()
                    print("||C|| = ", np.max(np.abs(model.C())))

                    model.evaluate_global()
                    RF_new = model.scatter_rhs()
                    if (np.abs(np.dot(delta, RF_new)) <= s * np.abs(np.dot(delta, RF)) or j == m):
                        RF = RF_new.copy()
                        break

        model.save_global_fields()
        model.update_plot()
        model.advance_model()

def halley_solve(model, num_steps, max_iters, tol):
    model.initialize_plot()
    for step in range(num_steps):
        print('Timestep', step)
        model.set_prescribed_dofs(step)
        model.compute_surf_tractions(step)
        for i in range(max_iters):
            model.set_global_fields()

            model.compute_local_state_variables()
            model.evaluate_local()

            model.evaluate_global()
            RF = model.scatter_rhs()
            print('Iteration: ', i)
            print(" ||C|| = ", np.max(np.abs(model.C())),
                  " ||R|| = ", np.linalg.norm(RF))
            if (np.linalg.norm(RF) < tol):
                break

            model.evaluate_tang()
            KFF = model.scatter_lhs()

            t1 = time.perf_counter()
            KFF_factorized = scipy.sparse.linalg.factorized(KFF)
            t2 = time.perf_counter()
            print("Factorizing stiffness: ", t2 - t1)
            delta = KFF_factorized(-RF)
            if (i > 1):
                model.set_newton_increment(delta)
                halley_rhs = model.evaluate_halley_correction()

                # # finite difference check for second derivative
                # halley_rhs_fd = model.evaluate_halley_correction_fd()
                # print(np.linalg.norm(halley_rhs - halley_rhs_fd))

                delta = delta ** 2 / (delta + 1 / 2 * KFF_factorized(halley_rhs))

            model.add_to_UF(delta)

            model.reset_xi()

        model.save_global_fields()
        model.update_plot()
        model.advance_model()

order = 2
problem = fem_problem("hole_block_traction", order, mixed=True)
num_steps, dt = problem.num_steps()

max_iters = 20
tol = 3e-10

model = Elastic_plastic_small(problem)
newton_solve(model, num_steps, max_iters, tol)

# # Save results as .xdmf file
# point_data, cell_data = model.get_data()
# problem.save_data("hole_block_displacement_2.xdmf", point_data, cell_data)
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, jacfwd

from cmad.fem_utils.global_deriv_types import GlobalDerivType

from abc import ABC
from cmad.fem_utils.fem_utils import assemble_global_fields, assemble_prescribed

from scipy.sparse import coo_matrix, csr_matrix

class Global_residual(ABC):
    def __init__(
            self, resid_fun, elem_surf_traction, volume_conn, elem_points,
            eq_num, params, num_nodes_elem, dof_node, num_free_dof, disp_node,
            disp_val, num_pres_dof, is_mixed, pres_surf_traction_points, pres_surf_traction,
            surf_traction_vector):

        self._global_resid = jit(vmap(resid_fun, in_axes=[0, None, 0]))
        self._global_jac = jit(vmap(jacfwd(resid_fun), in_axes=[0, None, 0]))
        self._surf_traction_batched = jit(vmap(elem_surf_traction, in_axes=[0, None]))

        self._deriv_mode = GlobalDerivType.DNONE

        self._num_free_dof = num_free_dof
        self._volume_conn = volume_conn
        self._elem_points = elem_points
        self._params = params
        self._eq_num = eq_num
        self._UF = np.zeros(num_free_dof)
        self._is_mixed = is_mixed

        # displacement and pressure boundary conditions
        self._disp_node = disp_node
        self._disp_val = disp_val
        self._num_pres_dof = num_pres_dof

        # traction boundary conditions
        self._pres_surf_traction_points = pres_surf_traction_points
        self._surf_traction_vector = surf_traction_vector
        self._pres_surf_traction = pres_surf_traction

        # indices for element vector assembly
        global_indices = eq_num[volume_conn, :].transpose(0, 2, 1).reshape(volume_conn.shape[0], -1)
        global_free_indices = np.where(global_indices > 0, global_indices - 1, -1)
        flat_global_free_indices = global_free_indices.ravel()
        mask_vector = flat_global_free_indices >= 0
        global_free_indices_vector = flat_global_free_indices[mask_vector]

        self._mask_vector = mask_vector
        self._global_free_indices_vector = global_free_indices_vector

        # indices for element matrix assembly
        elem_dofs = num_nodes_elem * dof_node
        ii, jj = np.meshgrid(np.arange(elem_dofs), np.arange(elem_dofs),
                                indexing='ij')
        row_f = global_free_indices[:, ii]
        col_f = global_free_indices[:, jj]
        mask_f = (row_f >= 0) & (col_f >= 0)
        row_f = row_f[mask_f]
        col_f = col_f[mask_f]

        self._ii = ii
        self._jj = jj
        self._row_f = row_f
        self._col_f = col_f
        self._mask_f = mask_f

        # index arrays for traction vector assembly
        if not pres_surf_traction is None:
            surf_global_indices_all = eq_num[:, :-1][pres_surf_traction, :]. \
                transpose(0, 2, 1).reshape(pres_surf_traction.shape[0], -1)
            flat_surf_global_free_indices = np.where(surf_global_indices_all > 0,
                                                    surf_global_indices_all - 1, -1).ravel()
            self._surf_valid_free_mask = flat_surf_global_free_indices >= 0
            self._surf_global_indices = flat_surf_global_free_indices[self._surf_valid_free_mask]

        # data storage
        self._point_data = []

    def evaluate(self):

        variables = self._variables()
        deriv_mode = self._deriv_mode

        if deriv_mode == GlobalDerivType.DNONE:
            self._R = np.asarray(self._global_resid(*variables))
            self._Jac = None
        elif deriv_mode == GlobalDerivType.DU:
            self._Jac = np.asarray(self._global_jac(*variables))

    def R(self):
        return self._R

    def Jac(self):
        return self._Jac

    def set_global_fields(self):
        UUR = assemble_global_fields(self._eq_num, self._UF, self._UP)
        self._u_elem = UUR[self._volume_conn, :].transpose(0, 2, 1) \
            .reshape(self._volume_conn.shape[0], -1)

    def save_global_fields(self):
        UUR = assemble_global_fields(self._eq_num, self._UF, self._UP)
        if self._is_mixed:
            self._point_data.append({'displacement_field': UUR[:, :-1],
                                    'pressure_field': UUR[:, -1]})
        else:
            self._point_data.append({'displacement_field': UUR})

    def compute_surf_tractions(self, step):
        self._FF = np.zeros(self._num_free_dof)
        if not self._pres_surf_traction is None:
            FEL = self._surf_traction_batched(self._pres_surf_traction_points,
                                                self._surf_traction_vector[step])
            np.add.at(self._FF, self._surf_global_indices, FEL.ravel()[self._surf_valid_free_mask])

    def get_data(self):
        return self._point_data

    def set_prescribed_dofs(self, step):
        self._UP = assemble_prescribed(self._num_pres_dof, self._disp_node,
                                       self._disp_val[:, step], self._eq_num)

    def _variables(self):
        return self._u_elem, self._params, self._elem_points

    def add_to_UF(self, delta):
        self._UF += delta

    def scatter_rhs(self):
        RF_global = np.zeros(self._num_free_dof)
        np.add.at(RF_global, self._global_free_indices_vector,
                  self._R.ravel()[self._mask_vector])
        return RF_global - self._FF

    def scatter_lhs(self):
        KFF = csr_matrix(coo_matrix((self._Jac[:, self._ii, self._jj][self._mask_f],
                                    (self._row_f, self._col_f)),
                                    shape=(self._num_free_dof, self._num_free_dof)))
        return KFF

    def seed_none(self):
        self._deriv_mode = GlobalDerivType.DNONE

    def seed_U(self):
        self._deriv_mode = GlobalDerivType.DU


















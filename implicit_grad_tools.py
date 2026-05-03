import numpy as np

def boundary_node_jacobian(control_6d, deltaL):
    """
    解析计算 J_map = d(q_b)/d(u) ，维度 8x6
    - u = [x_l, y_l, th_l, x_r, y_r, th_r]
    - q_b = [X0,Y0,X1,Y1,Xn_2,Yn_2,Xn_1,Yn_1]  (与你 env._control_node_from_xytheta 一致)

    返回:
      J_map: shape (8,6)
    """
    x_l, y_l, th_l, x_r, y_r, th_r = control_6d

    J = np.zeros((8, 6), dtype=float)

    # -----------------------------
    # 左端：
    # node0 = (x_l, y_l)
    # node1 = (x_l + cos(th_l)*dL, y_l + sin(th_l)*dL)
    # -----------------------------
    # X0, Y0
    J[0, 0] = 1.0  # dX0/dx_l
    J[1, 1] = 1.0  # dY0/dy_l

    # X1, Y1 w.r.t x_l, y_l
    J[2, 0] = 1.0
    J[3, 1] = 1.0

    # X1, Y1 w.r.t th_l
    J[2, 2] = -np.sin(th_l) * deltaL
    J[3, 2] =  np.cos(th_l) * deltaL

    # -----------------------------
    # 右端：
    # node_{n-2} = (x_r - cos(th_r)*dL, y_r - sin(th_r)*dL)
    # node_{n-1} = (x_r, y_r)
    # -----------------------------
    # Xn_2, Yn_2 w.r.t x_r, y_r
    J[4, 3] = 1.0
    J[5, 4] = 1.0

    # Xn_2, Yn_2 w.r.t th_r
    # Xn_2 = x_r - cos(th_r)*dL  -> d/dth = +sin(th_r)*dL
    # Yn_2 = y_r - sin(th_r)*dL  -> d/dth = -cos(th_r)*dL
    J[4, 5] =  np.sin(th_r) * deltaL
    J[5, 5] = -np.cos(th_r) * deltaL

    # Xn_1, Yn_1
    J[6, 3] = 1.0
    J[7, 4] = 1.0

    return J


def implicit_final_control_grad(J, free_index, fixed_index, a_free, control_6d, deltaL):
    """
    用伴随法做隐式梯度（只针对 final step）
    已知:
      - 平衡残差 r_f(q_f, q_b) = 0 的 Jacobian 子块来自 J=df/dq
      - a_free = dL/dq_free  (只在自由DOF上)
    返回:
      g_u : dL/du (6D)，对应 [x_l, y_l, th_l, x_r, y_r, th_r]

    公式：
      J_ff^T v = a_free
      g_qb = - J_fb^T v
      g_u  = (d q_b / d u)^T g_qb
    """
    # 取子块
    J_ff = J[np.ix_(free_index, free_index)]         # shape (nf,nf)
    J_fb = J[np.ix_(free_index, fixed_index)]        # shape (nf,8)

    # 解伴随
    v = np.linalg.solve(J_ff.T, a_free)              # shape (nf,)

    # dL/dq_b (8D)
    g_qb = - (J_fb.T @ v)                            # shape (8,)

    # map 8D -> 6D
    J_map = boundary_node_jacobian(control_6d, deltaL)  # (8,6)
    g_u = J_map.T @ g_qb                               # (6,)

    return g_u

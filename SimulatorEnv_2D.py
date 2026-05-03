# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

############################

def crossMat(a):
    A = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return A

def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the derivative of bending energy E_k^b with respect to
    x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dF : np.ndarray
        Derivative of bending energy.
    """

    ex = xk - xkm1
    ey = yk - ykm1
    fx = xkp1 - xk
    fy = ykp1 - yk

    norm_e = np.hypot(ex, ey)
    norm_f = np.hypot(fx, fy)

    tex = ex / norm_e
    tey = ey / norm_e
    tfx = fx / norm_f
    tfy = fy / norm_f

    chi = 1.0 + tex * tfx + tey * tfy
    if chi < 1e-8:
        chi = 1e-8  # Prevent division by zero
    kappa1 = 2.0 * (tex * tfy - tey * tfx) / chi
    ttx = (tex + tfx) / chi
    tty = (tey + tfy) / chi
    d2z = 2.0 / chi

    # Planar specialization of the original 3D cross-product formulas.
    Dkappa1De_x = (-kappa1 * ttx + tfy * d2z) / norm_e
    Dkappa1De_y = (-kappa1 * tty - tfx * d2z) / norm_e
    Dkappa1Df_x = (-kappa1 * ttx - tey * d2z) / norm_f
    Dkappa1Df_y = (-kappa1 * tty + tex * d2z) / norm_f

    # Populate the gradient of kappa
    gradKappa = np.array([
        -Dkappa1De_x,
        -Dkappa1De_y,
        Dkappa1De_x - Dkappa1Df_x,
        Dkappa1De_y - Dkappa1Df_y,
        Dkappa1Df_x,
        Dkappa1Df_y,
    ])

    # Gradient of bending energy
    dkappa = kappa1 - curvature0
    dF = gradKappa * EI * dkappa / l_k

    return dF

def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the Hessian (second derivative) of bending energy E_k^b
    with respect to x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dJ : np.ndarray
        Hessian of bending energy.
    """

    ex = xk - xkm1
    ey = yk - ykm1
    fx = xkp1 - xk
    fy = ykp1 - yk

    norm_e = np.hypot(ex, ey)
    norm_f = np.hypot(fx, fy)
    if norm_e < 1e-8 or norm_f < 1e-8:
        return np.zeros((6, 6))  # Prevent division by zero

    tex = ex / norm_e
    tey = ey / norm_e
    tfx = fx / norm_f
    tfy = fy / norm_f

    chi = 1.0 + tex * tfx + tey * tfy
    if chi < 1e-8:
        chi = 1e-8
    kappa1 = 2.0 * (tex * tfy - tey * tfx) / chi
    ttx = (tex + tfx) / chi
    tty = (tey + tfy) / chi
    d2z = 2.0 / chi

    tf_cross_d2_x = tfy * d2z
    tf_cross_d2_y = -tfx * d2z
    te_cross_d2_x = tey * d2z
    te_cross_d2_y = -tex * d2z

    # Gradient of kappa1 with respect to edge vectors.
    Dkappa1De_x = (-kappa1 * ttx + tf_cross_d2_x) / norm_e
    Dkappa1De_y = (-kappa1 * tty + tf_cross_d2_y) / norm_e
    Dkappa1Df_x = (-kappa1 * ttx - te_cross_d2_x) / norm_f
    Dkappa1Df_y = (-kappa1 * tty - te_cross_d2_y) / norm_f

    # Populate the gradient of kappa
    gradKappa = np.array([
        -Dkappa1De_x,
        -Dkappa1De_y,
        Dkappa1De_x - Dkappa1Df_x,
        Dkappa1De_y - Dkappa1Df_y,
        Dkappa1Df_x,
        Dkappa1Df_y,
    ])

    # Compute the Hessian (second derivative of kappa)
    DDkappa1 = np.zeros((6, 6))

    norm2_e = norm_e**2
    norm2_f = norm_f**2

    tt11 = ttx * ttx
    tt12 = ttx * tty
    tt22 = tty * tty
    tf_d2_tt11 = tf_cross_d2_x * ttx
    tf_d2_tt12 = tf_cross_d2_x * tty
    tf_d2_tt21 = tf_cross_d2_y * ttx
    tf_d2_tt22 = tf_cross_d2_y * tty
    te_d2_tt11 = te_cross_d2_x * ttx
    te_d2_tt12 = te_cross_d2_x * tty
    te_d2_tt21 = te_cross_d2_y * ttx
    te_d2_tt22 = te_cross_d2_y * tty
    chi_norm2_e = chi * norm2_e
    chi_norm2_f = chi * norm2_f
    inv_norm_e_norm_f = 1.0 / (norm_e * norm_f)
    mixed_kappa_scale = kappa1 / chi * inv_norm_e_norm_f

    # These are the top-left 2x2 blocks of the original 3D Hessian formulas.
    D2kappa1De2 = np.array([
        [(2 * kappa1 * tt11 - 2 * tf_d2_tt11) / norm2_e
         - kappa1 * (1.0 - tex * tex) / chi_norm2_e,
         (2 * kappa1 * tt12 - tf_d2_tt12 - tf_d2_tt21) / norm2_e
         + kappa1 * tex * tey / chi_norm2_e],
        [(2 * kappa1 * tt12 - tf_d2_tt12 - tf_d2_tt21) / norm2_e
         + kappa1 * tex * tey / chi_norm2_e,
         (2 * kappa1 * tt22 - 2 * tf_d2_tt22) / norm2_e
         - kappa1 * (1.0 - tey * tey) / chi_norm2_e],
    ])

    D2kappa1Df2 = np.array([
        [(2 * kappa1 * tt11 + 2 * te_d2_tt11) / norm2_f
         - kappa1 * (1.0 - tfx * tfx) / chi_norm2_f,
         (2 * kappa1 * tt12 + te_d2_tt12 + te_d2_tt21) / norm2_f
         + kappa1 * tfx * tfy / chi_norm2_f],
        [(2 * kappa1 * tt12 + te_d2_tt12 + te_d2_tt21) / norm2_f
         + kappa1 * tfx * tfy / chi_norm2_f,
         (2 * kappa1 * tt22 + 2 * te_d2_tt22) / norm2_f
         - kappa1 * (1.0 - tfy * tfy) / chi_norm2_f],
    ])

    D2kappa1DeDf = np.array([
        [-mixed_kappa_scale * (1.0 + tex * tfx)
         + inv_norm_e_norm_f * (2 * kappa1 * tt11 - tf_d2_tt11 + te_d2_tt11),
         -mixed_kappa_scale * (tex * tfy)
         + inv_norm_e_norm_f * (2 * kappa1 * tt12 - tf_d2_tt12 + te_d2_tt21 + d2z)],
        [-mixed_kappa_scale * (tey * tfx)
         + inv_norm_e_norm_f * (2 * kappa1 * tt12 - tf_d2_tt21 + te_d2_tt12 - d2z),
         -mixed_kappa_scale * (1.0 + tey * tfy)
         + inv_norm_e_norm_f * (2 * kappa1 * tt22 - tf_d2_tt22 + te_d2_tt22)],
    ])
    D2kappa1DfDe = D2kappa1DeDf.T

    # Populate the Hessian of kappa
    DDkappa1[0:2, 0:2] = D2kappa1De2
    DDkappa1[0:2, 2:4] = -D2kappa1De2 + D2kappa1DeDf
    DDkappa1[0:2, 4:6] = -D2kappa1DeDf
    DDkappa1[2:4, 0:2] = -D2kappa1De2 + D2kappa1DfDe
    DDkappa1[2:4, 2:4] = D2kappa1De2 - D2kappa1DeDf - D2kappa1DfDe + D2kappa1Df2
    DDkappa1[2:4, 4:6] = D2kappa1DeDf - D2kappa1Df2
    DDkappa1[4:6, 0:2] = -D2kappa1DfDe
    DDkappa1[4:6, 2:4] = D2kappa1DfDe - D2kappa1Df2
    DDkappa1[4:6, 4:6] = D2kappa1Df2

    # Hessian of bending energy
    dkappa = kappa1 - curvature0
    dJ = 1.0 / l_k * EI * np.outer(gradKappa, gradKappa)
    dJ += 1.0 / l_k * dkappa * EI * DDkappa1

    return dJ

def gradEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    Calculate the gradient of the stretching energy with respect to the coordinates.

    Args:
    - xk (float): x coordinate of the current point
    - yk (float): y coordinate of the current point
    - xkp1 (float): x coordinate of the next point
    - ykp1 (float): y coordinate of the next point
    - l_k (float): reference length
    - EA (float): elastic modulus

    Returns:
    - F (np.array): Gradient array
    """
    dx = xkp1 - xk
    dy = ykp1 - yk
    edge_len = np.hypot(dx, dy)
    if edge_len < 1e-12:
        return np.zeros(4)
    scale = EA * (1.0 / edge_len - 1.0 / l_k)

    return np.array([
        scale * dx,
        scale * dy,
        -scale * dx,
        -scale * dy,
    ])

def hessEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    This function returns the 4x4 Hessian of the stretching energy E_k^s with
    respect to x_k, y_k, x_{k+1}, and y_{k+1}.
    """
    dx = xkp1 - xk
    dy = ykp1 - yk
    r2 = dx * dx + dy * dy
    if r2 < 1e-24:
        return np.zeros((4, 4))
    edge_len = np.sqrt(r2)
    alpha = 1.0 / edge_len - 1.0 / l_k
    inv_r3 = 1.0 / (r2 * edge_len)

    B11 = EA * (alpha - dx * dx * inv_r3)
    B12 = EA * (-dx * dy * inv_r3)
    B22 = EA * (alpha - dy * dy * inv_r3)

    return np.array([
        [-B11, -B12, B11, B12],
        [-B12, -B22, B12, B22],
        [B11, B12, -B11, -B12],
        [B12, B22, -B12, -B22],
    ])

"""# Functions to create elastic force vector and its Hessian"""

def getFs(q, EA, deltaL):
  # q - DOF vector of size N
  # EA - stretching stiffness
  # deltaL - undeformed reference length (assume to be a scalar for this simple example)
  # Output:
  # Fs - a vector (negative gradient of elastic stretching force)
  # Js - a matrix (negative hessian of elastic stretching force)

  ndof = q.size # Number of DOFs
  N = ndof // 2 # Number of nodes

  Fs = np.zeros(ndof) # stretching force
  Js = np.zeros((ndof, ndof))

  for k in range(0, N-1):
      # May need to modify if network of beams
      # k-th stretching spring (USE A LOOP for the general case
      xkm1 = q[2*k] # x coordinate of the first node
      ykm1 = q[2*k+1] # y coordinate of the first node
      xk = q[2*k+2] # x coordinate of the second node
      yk = q[2*k+3] # y coordinate of the second node
      ind = slice(2*k, 2*k+4) # 0, 1, 2, 3 for k = 0
      gradEnergy = gradEs(xkm1, ykm1, xk, yk, deltaL, EA)
      hessEnergy = hessEs(xkm1, ykm1, xk, yk, deltaL, EA)

      Fs[ind] -= gradEnergy # force = - gradient of energy. Fs is the stretching force
      Js[ind, ind] -= hessEnergy # index vector: 0:4

  return Fs, Js

def getFb(q, EI, deltaL):
  # q - DOF vector of size N
  # EI - bending stiffness
  # deltaL - undeformed Voronoi length (assume to be a scalar for this simple example)
  # Output:
  # Fb - a vector (negative gradient of elastic stretching force)
  # Jb - a matrix (negative hessian of elastic stretching force)

  ndof = q.size # Number of DOFs
  N = ndof // 2 # Number of nodes

  Fb = np.zeros(ndof) # bending force
  Jb = np.zeros((ndof, ndof))

  # First bending spring (USE A LOOP for the general case)
  for k in range(1, N-1):
    xkm1 = q[2*k-2] # x coordinate of the first node
    ykm1 = q[2*k-1] # y coordinate of the first node
    xk = q[2*k] # x coordinate of the second node
    yk = q[2*k+1] # y coordinate of the second node
    xkp1 = q[2*k+2] # x coordinate of the third node
    ykp1 = q[2*k+3] # y coordinate of the third node
    ind = slice(2*k-2, 2*k+4)
    gradEnergy = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)
    hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)

    Fb[ind] -= gradEnergy # force = - gradient of energy. Fb is the stretching force
    Jb[ind, ind] -= hessEnergy # index vector: 0:6

  return Fb, Jb


def objfun(q_old, u_old, dt, tol, maximum_iter,
           m, mMat, EI, EA, W, deltaL, free_index,
           control_node=None,
           use_inertia=False):
    """
    单步隐式时间积分（Newton-Raphson）
    """
    if control_node is None:
        raise ValueError("Control node position must be provided.")

    q_new = q_old.copy()
    q_new[:4] = control_node[:4]  # left end nodes
    q_new[-4:] = control_node[4:]  # right end nodes

    iter_count = 0
    error = tol * 10
    flag = 1
    J = np.zeros((q_new.size, q_new.size))

    while error > tol:
        # inertia term should be 0 for quasi-static assumption
        if use_inertia:
            F_inertia = m / dt * ((q_new - q_old) / dt - u_old)
            J_inertia = mMat / dt ** 2
        else:
            F_inertia = np.zeros_like(q_new)
            J_inertia = np.zeros((q_new.size, q_new.size))

        Fs, Js = getFs(q_new, EA, deltaL)
        Fb, Jb = getFb(q_new, EI, deltaL)
        F_elastic = Fs + Fb
        J_elastic = Js + Jb

        f = F_inertia - F_elastic - W
        J = J_inertia - J_elastic
        f_free = f[free_index]
        J_free = J[np.ix_(free_index, free_index)]

        if not (np.all(np.isfinite(f_free)) and np.all(np.isfinite(J_free))):
            flag = -1
            break

        try:
            dq_free = np.linalg.solve(J_free, f_free)
        except np.linalg.LinAlgError:
            flag = -1
            break
        if not np.all(np.isfinite(dq_free)):
            flag = -1
            break
        q_new[free_index] = q_new[free_index] - dq_free
        if not np.all(np.isfinite(q_new)):
            flag = -1
            break

        error = np.linalg.norm(f_free)
        if not np.isfinite(error):
            flag = -1
            break
        iter_count += 1
        if iter_count > maximum_iter:
            flag = -1
            print("Maximum number of iterations reached.")
            break

    if not np.all(np.isfinite(q_new)):
        flag = -1
    return q_new, flag, J


############################
# SimulatorEnv_2D
############################

class SimulatorEnv_2D:
    """
    一个规范化的弹性杆仿真环境：
      - 控制量: (x, y, theta) 由外部给定，对应右端末端的位置与方向
      - 状态: 内部杆节点位置 q, 速度 u, 当前时间 ctime 等
      - 接口: reset(), step(control), render()
    """

    def __init__(self,
                 nv=51,
                 dt=0.01,
                #  dt=1.0,
                 rod_length=1.0,
                 total_time=10.0,
                 R_outer=0.013,
                 r_inner=0.011,
                 E_al=70e9,
                 rho_al=2700,
                 max_newton_iter=1000,
                 save_history=False):
        self.nv = nv
        self.dt = dt
        self.rod_length = rod_length
        self.total_time = total_time
        self.R_outer = R_outer
        self.r_inner = r_inner
        self.E_al = E_al
        self.rho_al = rho_al
        self.max_newton_iter = max_newton_iter
        self.save_history = save_history

        # 截面与材料参数
        self.I_sec = np.pi * (R_outer ** 4 - r_inner ** 4) / 4
        self.A_sec = np.pi * (R_outer ** 2 - r_inner ** 2)
        self.EI = E_al * self.I_sec
        self.EA = E_al * self.A_sec

        # 空间离散
        self.deltaL = rod_length / (nv - 1)

        # 节点坐标（初始直杆）
        self.nodes = np.zeros((nv, 2))
        for c in range(nv):
            self.nodes[c, 0] = c * self.deltaL
            self.nodes[c, 1] = 0.0

        # 质量
        m_node = np.pi * (R_outer ** 2 - r_inner ** 2) * rod_length * rho_al / (nv - 1)
        self.m = np.zeros(2 * nv)
        for k in range(nv):
            self.m[2 * k] = m_node
            self.m[2 * k + 1] = m_node
        self.mMat = np.diag(self.m)

        # 重力
        self.W = np.zeros(2 * nv)
        self.W[1::2] = -9.81 * m_node

        # 牛顿迭代容差
        self.tol = self.EI / rod_length ** 2 * 1e-3

        # 约束 DOF
        all_DOFs = np.arange(2 * nv)
        self.fixed_index = np.array([0, 1, 2, 3, 2 * nv - 4, 2 * nv - 3, 2 * nv - 2, 2 * nv - 1])
        self.free_index = np.setdiff1d(all_DOFs, self.fixed_index)

        # 中间节点索引
        # self.mid_node = nv // 2 + 1  # 与原脚本保持一致
        self.mid_node = nv // 2  # 与原脚本保持一致

        # 总步数
        self.Nsteps = int(round(total_time / dt))

        self.dlamda = 1 / self.Nsteps

        # 状态变量（在 reset 中初始化）
        self.q = None
        self.u = None
        self.ctime = None
        self.current_step = None
        self.lamda = None

        # 历史记录（可选）
        self.history = {
            "error": [],
            "control": [],
            "midx": [],
            "midy": []
        }

    # ---------- inner tools ----------

    def _control_node_from_xytheta(self, control_vec):

        """
          node_0 = (x_l, y_l)
          node_1 = (x_l + np.cos(th_l)*self.deltaL, y_l + np.sin(th_l)*self.deltaL)
          ...
          node_{N-1} = (x_r, y_r)
          node_{N-2} = (x_r - np.cos(th_r)*self.deltaL, y_r - np.sin(th_r)*self.deltaL)
        order: [node_0_x, node_0_y, node_1_x, node_1_y, ..., node_{N-2}_x, node_{N-2}_y, node_{N-1}_x, node_{N-1}_y]
        """
        x_l, y_l, th_l, x_r, y_r, th_r = control_vec
        
        X_0 = x_l
        Y_0 = y_l
        X_1 = x_l + np.cos(th_l) * self.deltaL
        Y_1 = y_l + np.sin(th_l) * self.deltaL
        Xn_2 = x_r - np.cos(th_r) * self.deltaL
        Yn_2 = y_r - np.sin(th_r) * self.deltaL
        Xn_1 = x_r
        Yn_1 = y_r
        return np.array([X_0, Y_0, X_1, Y_1, Xn_2, Yn_2, Xn_1, Yn_1], dtype=float)

    def _get_midpoint(self):
        """返回当前中点的坐标 (midx, midy)。"""
        x_arr = self.q[::2]
        y_arr = self.q[1::2]
        midx = x_arr[self.mid_node]
        midy = y_arr[self.mid_node]
        return midx, midy

    def _get_observation(self):
        """观测量仅包含中点坐标。"""
        midx, midy = self._get_midpoint()
        return {
            "time": self.ctime,
            "midx": midx,
            "midy": midy
        }

    # ---------- API ----------

    def reset(self,
              init_x=None,
              init_y=None,
              init_theta=None):
        """
        重置环境，返回初始观测。
        控制量初值只影响你第一步时施加的控制，这里默认末端在 (rod_length, 0, 0)。
        """
        if init_x is None:
            init_x = self.rod_length
        if init_y is None:
            init_y = 0.0
        if init_theta is None:
            init_theta = 0.0

        self.q = np.zeros(2 * self.nv)
        for c in range(self.nv):
            self.q[2 * c] = self.nodes[c, 0]
            self.q[2 * c + 1] = self.nodes[c, 1]

        self.u = np.zeros_like(self.q)
        self.ctime = 0.0
        self.current_step = 0
        self.lamda = 0.0

        # 清空历史
        if self.save_history:
            for k in self.history:
                self.history[k] = []

        # 不在 reset 里做一步积分，只返回当前观测
        return self._get_observation()

    def step(self, control, use_inertia=False):
        """
        单步仿真：
          control: (x, y, theta)，numpy 数组或可迭代
        返回:
          obs, reward, done, info
        """
        control = np.asarray(control, dtype=float).flatten()
        if control.shape[0] != 6:
            raise ValueError("control must be a 6D vector: [x_l, y_l, th_l, x_r, y_r, th_r].")

        # x, y, theta = control
        control_node = self._control_node_from_xytheta(control)

        # 调用隐式积分器
        q_new, flag, J = objfun(
            q_old=self.q,
            u_old=self.u,
            dt=self.dt,
            tol=self.tol,
            maximum_iter=self.max_newton_iter,
            m=self.m,
            mMat=self.mMat,
            EI=self.EI,
            EA=self.EA,
            W=self.W,
            deltaL=self.deltaL,
            free_index=self.free_index,
            control_node=control_node,
            use_inertia=use_inertia,
        )

        # 更新速度、时间
        u_new = (q_new - self.q) / self.dt
        self.q = q_new
        self.u = u_new
        self.ctime += self.dt
        self.current_step += 1
        self.lamda = self.current_step * self.dlamda

        # 观测与误差
        obs = self._get_observation()
        done = (self.current_step >= self.Nsteps) or (flag < 0)

        info = {
            "converged": (flag > 0),
            "control": control,
            "q": self.q.copy(),
            "u": self.u.copy(),
            "J": J,
            "free_index": self.free_index.copy(),
            "fixed_index": self.fixed_index.copy(),
            "control_node": control_node.copy(),  # 8D 边界DOF
        }

        # 保存历史
        if self.save_history:
            self.history["control"].append(control.copy())
            self.history["midx"].append(obs["midx"])
            self.history["midy"].append(obs["midy"])

        return obs, done, info

    def render(self, saveimg = False, show_target=True, show=True, ax=None):
        """
        可视化当前杆的形状。
        """
        x_arr = self.q[::2]
        y_arr = self.q[1::2]
        midx, midy = self._get_midpoint()

        # 第一次调用时初始化 figure 和 axis
        if not hasattr(self, "_render_initialized") or not self._render_initialized:
            self.fig, self.ax = plt.subplots()
            self._render_initialized = True
            plt.ion()  # 开启交互模式

        ax = self.ax
        ax.cla()
        ax.plot(x_arr, y_arr, 'ko-', label="rod")
        ax.plot(midx, midy, 'ro', markersize=8, label='midpoint')
        # if show_target:
        #     ax.plot(midx_target, midy_target, 'bx', markersize=10, label='target')

        ax.set_title(f"t = {self.ctime:.3f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis('equal')
        ax.set_xlim(-0.2, self.rod_length + 0.3)
        ax.set_ylim(-1.0, 1.0)
        ax.legend()

        if saveimg:
            plt.savefig(f'imgs/render_{self.ctime:.3f}.png')
            print(f"Saved image: imgs/render_{self.ctime:.3f}.png")

        if show:
            plt.pause(0.01)

if __name__ == "__main__":
    # 简单测试环境
    env = SimulatorEnv_2D(nv=31, dt=0.01, rod_length=1.0, total_time=10.0, save_history=False)
    obs = env.reset()

    done = False
    while not done:
        t = obs["time"]
        x_target = env.rod_length
        y_target = 0.1 * np.sin(2 * np.pi * 0.5 * t)
        theta_target = 0.0
        control = np.array([0.0, - y_target, theta_target, x_target, y_target, theta_target])

        obs, done, info = env.step(control)
        env.render(show=True)

    print("Simulation finished.")

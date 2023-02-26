from sympy import *
from sympy.printing.numpy import JaxPrinter

# determines that whether we consider the feedback controller when linearizing the system
CONSIDER_FEEDBACK_CONTROL = True

jax_printer = JaxPrinter()

m1, m2, j1, j2 = symbols("m1 m2 j1 j2")
l1, l2, lc1, lc2 = symbols("l1 l2 lc1 lc2")
sym_g = symbols("g")
th1, th2 = symbols("th1 th2")
th_d1, th_d2 = symbols("th_d1 th_d2")
tau1, tau2 = symbols("tau1 tau2")
th_des1, th_des2 = symbols("th_des1 th_des2")
th_d_des1, th_d_des2 = symbols("th_d_des1 th_d_des2")
kp1, kp2, kd1, kd2 = symbols("kp1 kp2, kd1, kd2")
latex_symbol_names = {
    m1: "m_1",
    m2: "m_2",
    j1: "J_1",
    j2: "J_2",
    l1: "l_1",
    l2: "l_2",
    lc1: "l_{c1}",
    lc2: "l_{c2}",
    sym_g: "g",
    th1: "\\theta_1",
    th2: "\\theta_2",
    th_d1: "\\dot{\\theta}_1",
    th_d2: "\\dot{\\theta}_2",
    tau1: "\\tau_1",
    tau2: "\\tau_2",
    th_des1: "\\theta_1^\\mathrm{d}",
    th_des2: "\\theta_2^\\mathrm{d}",
    th_d_des1: "\\dot{\\theta}_1^\\mathrm{d}",
    th_d_des2: "\\dot{\\theta}_2^\\mathrm{d}",
    kp1: "k_{p,1}",
    kp2: "k_{p,2}",
    kd1: "k_{d,1}",
    kd2: "k_{d,2}",
}

# group symbolic states into matrices
th = Matrix([th1, th2])
th_d = Matrix([th_d1, th_d2])
tau = Matrix([tau1, tau2])
th_des = Matrix([th_des1, th_des2])
th_d_des = Matrix([th_d_des1, th_d_des2])
kp = Matrix([[kp1, 0], [0, kp2]])
kd = Matrix([[kd1, 0], [0, kd2]])

# define gravity vector with gravity acting in negative y direction
g = Matrix([0, -sym_g])

# Position of elbow joint and end-effector
x_eb = Matrix([l1 * cos(th1), l1 * sin(th1)])
x = x_eb + Matrix([l2 * cos(th2), l2 * sin(th2)])

# Position of Center of Mass (CoM) of link 1 and link 2
x_c1 = Matrix([lc1 * cos(th1), lc1 * sin(th1)])
x_c2 = x_eb + Matrix([lc2 * cos(th2), lc2 * sin(th2)])

# Rotation matrix from base to link 1 and link 2
R1 = Matrix([[cos(th1), -sin(th1)], [sin(th1), cos(th1)]])
R2 = Matrix([[cos(th2), -sin(th2)], [sin(th2), cos(th2)]])

# Positional Jacobian of elbow joint and end-effector
J_eb = x_eb.jacobian(th)
J = x.jacobian(th)

# Positional Jacobian of the center of mass of the first link and second link
J_xc1 = x_c1.jacobian(th)
J_xc2 = x_c2.jacobian(th)
# Rotational Jacobian of the first link and second link
J_O1 = Identity(2)
J_O2 = Identity(2)

# time-derivative of the positional Jacobian of end-effector
J_dot = J.diff(th1) * th_d1 + J.diff(th2) * th_d2

# compute the mass matrix M
M = J_xc1.T * m1 * J_xc1 + J_xc2.T * m2 * J_xc2 + Matrix([[j1, 0], [0, j2]])
M = simplify(M)
print("M = ", latex(M, symbol_names=latex_symbol_names))

# compute the Coriolis and Centrifugal matrix C using Christoffel symbols
# Siciliano et al. (2009) p. 258, eq. 7.45
c111 = 0.5 * diff(M[0, 0], th1)
c112 = 0.5 * (diff(M[0, 0], th2) + diff(M[0, 1], th1) - diff(M[0, 1], th1))
c121 = 0.5 * (diff(M[0, 1], th1) + diff(M[0, 0], th2) - diff(M[1, 0], th1))
c122 = 0.5 * (diff(M[0, 1], th2) + diff(M[0, 1], th2) - diff(M[1, 1], th1))
c211 = 0.5 * (diff(M[1, 0], th1) + diff(M[1, 0], th1) - diff(M[0, 0], th2))
c212 = 0.5 * (diff(M[1, 0], th2) + diff(M[1, 1], th1) - diff(M[0, 1], th2))
c221 = 0.5 * (diff(M[1, 1], th1) + diff(M[1, 0], th2) - diff(M[1, 0], th2))
c222 = 0.5 * diff(M[1, 1], th2)
# eq. 7.44
c11 = c111 * th_d1 + c112 * th_d2
c12 = c121 * th_d1 + c122 * th_d2
c21 = c211 * th_d1 + c212 * th_d2
c22 = c221 * th_d1 + c222 * th_d2
C = simplify(Matrix([[c11, c12], [c21, c22]]), rational=True)
print("C = ", latex(C, symbol_names=latex_symbol_names))

# computation of potential energy
# gravity vector
U = -m1 * g.T * x_c1 - m2 * g.T * x_c2
G = simplify(U.jacobian(th).transpose())
print("G = ", latex(G, symbol_names=latex_symbol_names))

# state in decoupled form of dynamics
x = Matrix([th1, th2, th_d1, th_d2])
# output in decoupled form
y = Matrix([th1, th2])

# continuous dynamics in decoupled form
if CONSIDER_FEEDBACK_CONTROL is True:
    # closed-loop continuous dynamics in decoupled form
    th_dd = M.inv() * (tau + kp * (th_des - th) + kd * (th_d_des - th_d) - C * th_d - G)
    dx_dt = th_d.col_join(th_dd)
else:
    # open-loop dynamics in decoupled form
    dx_dt = th_d.col_join(M.inv() * (tau - C * th_d - G))

# derivation of state space matrices using linearization
print("Linearizing the decoupled dynamics using Taylor expansion ...")
A = simplify(dx_dt.jacobian(x))
B = simplify(dx_dt.jacobian(tau))
C = simplify(y.jacobian(x))
D = simplify(y.jacobian(tau))

print("A = ", latex(A, symbol_names=latex_symbol_names))
print("B = ", latex(B, symbol_names=latex_symbol_names))
print("C = ", latex(C, symbol_names=latex_symbol_names))
print("D = ", latex(D, symbol_names=latex_symbol_names))

print("Converting to JAX code ...")
print("A = ")
print(jax_printer.doprint(A))
print("B = ")
print(jax_printer.doprint(B))
print("C = ")
print(jax_printer.doprint(C))
print("D = ")
print(jax_printer.doprint(D))

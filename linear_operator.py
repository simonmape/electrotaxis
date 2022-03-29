from dolfin import *
import math
from fenics import *
from scipy.integrate import odeint
import numpy as np
from ufl import nabla_div
from tqdm import tqdm
import sys

# Simulator settings
num_points = 70
mesh = RectangleMesh(Point(0.0, 20.0), Point(60, 80), num_points, num_points)

W = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, "CG", 2)
TS = TensorFunctionSpace(mesh, 'P', 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)

# Make mixed spaces
phasespace_element = P1 * P1
polarityspace_element = P2 * P2
flowspace_element = P2 * P1

phasespace = FunctionSpace(mesh, phasespace_element)
polarityspace = FunctionSpace(mesh, polarityspace_element)
flowspace = FunctionSpace(mesh, flowspace_element)

polarity_assigner = FunctionAssigner(V, V)
stress_assigner = FunctionAssigner(TS, TS)
velocity_assigner = FunctionAssigner(flowspace.sub(0), V)
pressure_assigner = FunctionAssigner(flowspace.sub(1), W)
phi_assigner = FunctionAssigner(W, W)
velocity_assigner_inv = FunctionAssigner(V, flowspace.sub(0))
pressure_assigner_inv = FunctionAssigner(W, flowspace.sub(1))

# Set fenics parameters
parameters["form_compiler"]["quadrature_degree"] = 3
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["krylov_solver"]["nonzero_initial_guess"] = True
set_log_level(20)

# Set fixed simulation parameters
k = 1
alpha = a = 1
eta = 5. / 3.
kappa = 0.04
xi = 1.1
w_sa = 1
phi0 = 2
phicr = 1
D = 0.007
M = D / a
gamma = 0.04
zeta = 0.01
a = 1
# cE=1
Gamma = 1.
dt = 0.01
min_angle = 0.05
minphi = 0.5
minphi_b = 0.25
numSteps = int(10 / dt)  # electrotaxis time is 10 hours
U = 3600

# Set simulation parameters we do inference on
cE = 0.3
beta = 0.4
delta_ph = 0

# Define main expressions
class pIC(UserExpression):
    def eval(self, value, x):
        value[:] = [0, 0]

    def value_shape(self):
        return (2,)

class vIC(UserExpression):
    def eval(self, value, x):
        value[:] = [0, 0]

    def value_shape(self):
        return (2,)

class phiIC(UserExpression):
    def eval(self, value, x):
        value[:] = 0
        if 10 < x[0] < 20 and 45 < x[1] < 55:
            value[:] = 1

    def value_shape(self):
        return ()

class scalarZero(UserExpression):
    def eval(self, value, x):
        value[:] = 0

    def value_shape(self):
        return ()

class fixedField(UserExpression):
    def eval(self, value, x):
        value[:] = [1, 0]

    def value_shape(self):
        return (2,)

class pointRight(UserExpression):
    def eval(self, value, x):
        value[:] = [1, 0]

    def value_shape(self):
        return (2,)

class pointUp(UserExpression):
    def eval(self, value, x):
        value[:] = [0, 1]

    def value_shape(self):
        return (2,)



boundary = 'near(x[1],20) || near(x[1], 80) || near(x[0], 0) || near(x[0],60)'
n = FacetNormal(mesh)

#FLOW PROBLEM
vpr_new = Function(flowspace)
v_new, pr_new = split(vpr_new)

yw = TestFunction(flowspace)
y, w = split(yw)

dU = TrialFunction(flowspace)
(du1, du2) = split(dU)

a_v = eta*inner(nabla_grad(du1), nabla_grad(y)) * dx + \
      gamma * inner(du1, y) * dx + dot(nabla_grad(du2), y) * dx + \
      div(du1) * w * dx
zero = Expression(('0.0', '0.0', '0.0'), degree=2)
bcs_flow = DirichletBC(flowspace, zero, boundary)

#POLARITY PROBLEM
pols_new = Function(polarityspace)
p_new, pder_new = split(pols_new)
yz = TestFunction(polarityspace)
yp, zp = split(yz)

dP = TrialFunction(polarityspace)
(dp1, dp2) = split(dP)

a_pol = (1. / dt) * dot(dp1,yp) * dx + (1. / Gamma) * inner(dp2, yp) * dx + \
       inner(dp2, zp) * dx - kappa * inner(nabla_grad(dp1), nabla_grad(zp)) * dx
zero = Expression(('0.0', '0.0', '0.0','0.0'), degree=2)
bcs_pol = DirichletBC(polarityspace, zero, boundary)

# PHASE FIELD PROBLEM#
phis_new = Function(phasespace)
phi_new, phider_new = split(phis_new)
w1w2 = TestFunction(phasespace)
w1, w2 = split(w1w2)
dphi = TrialFunction(phasespace)
(dphi1, dphi2) = split(dphi)
a_phi = (1. / dt) * dphi1 * w1 * dx + M * dot(nabla_grad(dphi2), nabla_grad(w1)) * dx +\
        dphi2 * w2 * dx - k * dot(nabla_grad(dphi1), nabla_grad(w2)) * dx
zero = Expression(('0.0','0.0'), degree=2)
bcs_phi = DirichletBC(phasespace, zero, boundary)

#Set initial conditions
v_old = interpolate(vIC(), V)
pr_old = TestFunction(W)  # interpolate(scalarZero(),W)

# Assign initial conditions for phi fields
phi_old = interpolate(phiIC(), W)
phider_old = TestFunction(W)  # interpolate(scalarZero(),W)

# Assign initial conditions for polarity fields
p_old = interpolate(pIC(), V)
pder_old = TestFunction(V)  # interpolate(vIC(),V)

# Define the functions to be loaded here
scalar_space = FunctionSpace(mesh, P1)
E = interpolate(fixedField(), V)
right = interpolate(pointRight(), V)
up = interpolate(pointUp(), V)
sumstat = np.zeros((numSteps, 10))

for i in tqdm(range(numSteps)):
    t = i * dt

    # molecular field evolution
    if t < 1 or t > 4:
        field = interpolate(vIC(), V)
    else:
        field = interpolate(pointRight(), V)

    #VELOCITY
    L_v = inner(outer(p_old, p_old), nabla_grad(y)) * dx
    solve(a_v == L_v, vpr_new, bcs_flow, solver_parameters=dict(linear_solver='gmres',
                                                                      preconditioner='ilu'))
    # POLARITY EVOLUTION #
    L_pol = (1. / dt) * dot(p_old, yp) * dx - inner(nabla_grad(p_old) * (v_new + w_sa * p_old), yp) * dx - \
            (alpha / phicr) * inner((phi_old - phicr) * p_old, zp) * dx + \
            dot(p_old, p_old) * alpha * inner(p_old, zp) * dx - \
            cE * (1 + delta_ph * inner(nabla_grad(phi_old), nabla_grad(phi_old)) / (
                1 + inner(nabla_grad(phi_old), nabla_grad(phi_old)))) * inner(field, zp) * dx + \
            beta * inner(nabla_grad(phi_old), zp) * dx
    solve(a_pol == L_pol, pols_new, bcs_pol, solver_parameters=dict(linear_solver='gmres',
                                                                 preconditioner='ilu'))

    # PHASE FIELD PROBLEM#
    L_phi = (1. / dt) * phi_old * w1 * dx - div(phi_old * (v_new + w_sa * p_old)) * w1 * dx + \
            (a / (2 * phicr ** 4)) * phi_old * (phi_old - phi0) * (2 * phi_old - phi0) * w2 * dx - \
            (alpha / (2 * phicr)) * dot(p_old, p_old) * w2 * dx - \
            beta * div(p_new) * w2 * dx
    solve(a_phi == L_phi, phi_new, bcs_phi, solver_parameters=dict(linear_solver='gmres',
                                                                      preconditioner='ilu'))

    # ASSIGN ALL VARIABLES FOR NEW STEP
    str_old.assign(str_new)
    p_old.assign(p_new)
    velocity_assigner_inv.assign(v_old, vpr_new.sub(0))
    phi_old.assign(phi_new)
    pressure_assigner_inv.assign(pr_old, vpr_new.sub(1))
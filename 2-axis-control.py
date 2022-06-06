from dolfin import *
import math
from fenics import *
from scipy.integrate import odeint
import numpy as np
from ufl import nabla_div
from tqdm import tqdm
import sys
import os

# Simulator settings
num_points = 100
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

velocity_assigner_inv = FunctionAssigner(V, flowspace.sub(0))
pressure_assigner_inv = FunctionAssigner(W, flowspace.sub(1))
polarity_assigner_inv = FunctionAssigner(V, polarityspace.sub(0))
polder_assigner_inv = FunctionAssigner(V, polarityspace.sub(0))
phi_assigner_inv = FunctionAssigner(W, phasespace.sub(0))
phider_assigner_inv = FunctionAssigner(W, phasespace.sub(0))

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
Gamma = 0.125
dt = 0.01
numSteps = int(7 / dt)
U = 3600

# Set simulation parameters we do inference on
cE = Constant(0.015)
delta_ph = Constant(-0.4)
thres = 0.5
beta_begin = 0.018
beta_final = 0.16

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

class bulk(UserExpression):
    def eval(self, value, x):
        value[:] = 0
        if 13 < x[0] < 17 and 48 < x[1] < 52:
            value[:] = 1

    def value_shape(self):
        return ()

class edge(UserExpression):
    def eval(self, value, x):
        value[:] = 0
        if (10 < x[0] < 20 and 45 < x[1] < 47) or (10 < x[0] < 20 and 53 < x[1] < 55) or (10 < x[0] < 12 and 45 < x[1] < 55) or (18 < x[0] < 20 and 45 < x[1] < 55):
            value[:] = 1
    def value_shape(self):
        return ()

boundary = 'near(x[1],20) || near(x[1], 80) || near(x[0], 0) || near(x[0],60)'

# FLOW PROBLEM
vpr_new = Function(flowspace)
v_new, pr_new = split(vpr_new)

yw = TestFunction(flowspace)
y, w = split(yw)

dU = TrialFunction(flowspace)
(du1, du2) = split(dU)

a_v = eta * inner(nabla_grad(du1), nabla_grad(y)) * dx + \
      gamma * inner(du1, y) * dx + dot(nabla_grad(du2), y) * dx + \
      div(du1) * w * dx
zero = Expression(('0.0', '0.0', '0.0'), degree=2)
bcs_flow = DirichletBC(flowspace, zero, boundary)

# POLARITY PROBLEM
pols_new = Function(polarityspace)
p_new, pder_new = split(pols_new)
yz = TestFunction(polarityspace)
yp, zp = split(yz)

dP = TrialFunction(polarityspace)
(dp1, dp2) = split(dP)

a_pol = (1. / dt) * dot(dp1, yp) * dx + (1. / Gamma) * inner(dp2, yp) * dx + \
        inner(dp2, zp) * dx - kappa * inner(nabla_grad(dp1), nabla_grad(zp)) * dx
zero = Expression(('0.0', '0.0', '0.0', '0.0'), degree=2)
bcs_pol = DirichletBC(polarityspace, zero, boundary)

# PHASE FIELD PROBLEM#
phis_new = Function(phasespace)
phi_new, phider_new = split(phis_new)
w1w2 = TestFunction(phasespace)
w1, w2 = split(w1w2)
dphi = TrialFunction(phasespace)
(dphi1, dphi2) = split(dphi)
a_phi = (1. / dt) * dphi1 * w1 * dx + M * dot(nabla_grad(dphi2), nabla_grad(w1)) * dx + \
        dphi2 * w2 * dx - k * dot(nabla_grad(dphi1), nabla_grad(w2)) * dx
zero = Expression(('0.0', '0.0'), degree=2)
bcs_phi = DirichletBC(phasespace, zero, boundary)

# Region problem
w_region = TestFunction(W)
region_trial = TrialFunction(W)
a_region = (1. / dt) * region_trial * w_region * dx
zero = Expression('0.0', degree=2)
bcs_region = DirichletBC(W, zero, boundary)

# Set initial conditions
v_old = interpolate(vIC(), V)
pr_old = TestFunction(W)  # interpolate(scalarZero(),W)

# Assign initial conditions for phi fields
phi_old = interpolate(phiIC(), W)
phider_old = TestFunction(W)  # interpolate(scalarZero(),W)

# Assign initial conditions for polarity fields
p_old = interpolate(pIC(), V)
pder_old = TestFunction(V)  # interpolate(vIC(),V)

# Assign initial conditions for location field
bulk_region_old, bulk_region = interpolate(bulk(), W), Function(W)
edge_old, edge_region = interpolate(edge(),W), Function(W)

# Define the functions to be loaded here
scalar_space = FunctionSpace(mesh, P1)
E = interpolate(fixedField(), V)
right = interpolate(pointRight(), V)
up = interpolate(pointUp(), V)
sumstat = np.zeros((numSteps, 3))

beta = Constant(beta_begin)
field = Expression(('A', 'B'), A=Constant(0), B=Constant(0), degree=2)
# VELOCITY
L_v = zeta * inner(outer(p_old, p_old), nabla_grad(y)) * dx

# POLARITY EVOLUTION #
L_pol = (1. / dt) * dot(p_old, yp) * dx - inner(nabla_grad(p_old) * (v_new + w_sa * p_old), yp) * dx - \
        (alpha / phicr) * inner((phi_old - phicr) * p_old, zp) * dx + \
        dot(p_old, p_old) * alpha * inner(p_old, zp) * dx - \
        cE * (1 + delta_ph * edge_old) * inner(field, zp) * dx + \
        beta * inner(nabla_grad(phi_old), zp) * dx

# PHASE FIELD PROBLEM#
L_phi = (1. / dt) * phi_old * w1 * dx - div(phi_old * (v_new + w_sa * p_old)) * w1 * dx + \
        (a / (2 * phicr ** 4)) * phi_old * (phi_old - phi0) * (2 * phi_old - phi0) * w2 * dx - \
        (alpha / (2 * phicr)) * dot(p_old, p_old) * w2 * dx - \
        beta * div(p_new) * w2 * dx

# Move bulk region
L_bulk = (1. / dt) * bulk_region_old * w_region * dx - div(bulk_region_old * (v_new + w_sa * p_old)) * w_region * dx
L_edge = (1. / dt) * edge_old * w_region * dx - div(edge_old * (v_new + w_sa * p_old)) * w_region * dx
for i in tqdm(range(numSteps)):
    t = i * dt
    # Beta increases in this problem
    beta.assign(beta_begin + ((t / 12) ** 2) * (beta_final - beta_begin))

    # molecular field evolution
    if t < 3:
        field.A = Constant(0)
        field.B = Constant(0)
    elif t >= 3 and t<5:
        field.A = Constant(1)
        field.B = Constant(0)
    elif t >= 5:
        field.A = Constant(0)
        field.B = Constant(1)

    solve(a_v == L_v, vpr_new, bcs_flow, solver_parameters=dict(linear_solver='superlu_dist',
                                                                preconditioner='ilu'))

    solve(a_pol == L_pol, pols_new, bcs_pol, solver_parameters=dict(linear_solver='superlu_dist',
                                                                    preconditioner='ilu'))

    solve(a_phi == L_phi, phis_new, bcs_phi, solver_parameters=dict(linear_solver='superlu_dist',
                                                                    preconditioner='ilu'))

    solve(a_region == L_bulk, bulk_region, bcs_region, solver_parameters=dict(linear_solver='superlu_dist',
                                                                              preconditioner='ilu'))
    solve(a_region == L_edge, edge_region, bcs_region, solver_parameters=dict(linear_solver='superlu_dist',
                                                                              preconditioner='ilu'))
     # ASSIGN ALL VARIABLES FOR NEW STEP
    polarity_assigner_inv.assign(p_old, pols_new.sub(0))
    velocity_assigner_inv.assign(v_old, vpr_new.sub(0))
    phi_assigner_inv.assign(phi_old, phis_new.sub(0))
    bulk_region_old.assign(bulk_region)

    # Compute bulk directionality and speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: bulk_region_old(x) > thres)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))

    try:
        sumstat[i, 0] = assemble((inner(p_old, right) / sqrt(inner(p_old, p_old))) * dx_sub(1)) / area
        sumstat[i, 1] = assemble((inner(p_old, up) / sqrt(inner(p_old, p_old))) * dx_sub(1)) / area
        sumstat[i, 2] = assemble(abs(270 * p_old[0] + v_old[0]) * dx_sub(1)) / area

    except Exception as e:
        print('bulk', i, e)
    print('X dir:', sumstat[i, 0], 'Y dir:', sumstat[i, 1], 'Vel:',sumstat[i, 2])

np.savetxt('2axis.txt', sumstat)

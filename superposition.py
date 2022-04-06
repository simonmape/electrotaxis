from dolfin import *
import math
from fenics import *
from scipy.integrate import odeint
import numpy as np
from ufl import nabla_div
from tqdm import tqdm
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"

# Simulator settings
num_points = 60
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
Gamma = 1./2.
dt = 0.05
numSteps = int(10 / dt)
U = 3600

# Set simulation parameters we do inference on
cE = 0.3
beta = 0.4
delta_ph = -0.5
f_field = float(sys.argv[1])

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

class right(UserExpression):
    def eval(self, value, x):
        value[:] = 0
        if 18 < x[0] < 20 and 47 < x[1] < 53:
            value[:] = 1
    def value_shape(self):
        return ()

class left(UserExpression):
    def eval(self, value, x):
        value[:] = 0
        if 10 < x[0] < 12 and 47 < x[1] < 53:
            value[:] = 1
    def value_shape(self):
        return ()

class top(UserExpression):
    def eval(self, value, x):
        value[:] = 0
        if 12 < x[0] < 18 and 53 < x[1] < 55:
            value[:] = 1
    def value_shape(self):
        return ()

class bottom(UserExpression):
    def eval(self, value, x):
        value[:] = 0
        if 12 < x[0] < 18 and 45 < x[1] < 47:
            value[:] = 1
    def value_shape(self):
        return ()

class bulk(UserExpression):
    def eval(self, value, x):
        value[:] = 0
        if 12 < x[0] < 18 and 47 < x[1] < 53:
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

#Region problem
w_region = TestFunction(W)
region_trial = TrialFunction(W)
a_region = (1. / dt) * region_trial * w_region * dx
zero = Expression('0.0', degree=2)
bcs_region = DirichletBC(W, zero, boundary)

#Set initial conditions
v_old = interpolate(vIC(), V)
pr_old = TestFunction(W)  # interpolate(scalarZero(),W)

# Assign initial conditions for phi fields
phi_old = interpolate(phiIC(), W)
phider_old = TestFunction(W)  # interpolate(scalarZero(),W)

# Assign initial conditions for polarity fields
p_old = interpolate(pIC(), V)
pder_old = TestFunction(V)  # interpolate(vIC(),V)

# Assign initial conditions for location fields
leading_edge_old, leading_edge = interpolate(right(),W), Function(W)
trailing_edge_old, trailing_edge = interpolate(left(),W), Function(W)
top_edge_old, top_edge = interpolate(top(),W), Function(W)
bottom_edge_old, bottom_edge = interpolate(bottom(),W), Function(W)
bulk_region_old, bulk_region = interpolate(bulk(),W), Function(W)
edge_old, edge_region = interpolate(edge(),W), Function(W)

# Define the functions to be loaded here
scalar_space = FunctionSpace(mesh, P1)
E = interpolate(fixedField(), V)
right = interpolate(pointRight(), V)
up = interpolate(pointUp(), V)
sumstat = np.zeros((numSteps, 10))

timeseries_phi = TimeSeries('results/phi_superposition_insensitive_1_-5.txt')
timeseries_p = TimeSeries('results/p_superposition_insensitive_1_-5.txt')
timeseries_v = TimeSeries('results/v_superposition_insensitive_1_-5.txt')
timeseries_edge = TimeSeries('results/edge_superposition_insensitive_1_-5.txt')
timeseries_bulk = TimeSeries('results/bulk_superposition_insensitive_1_-5.txt')

for i in tqdm(range(numSteps)):
    t = i * dt

    timeseries_phi.store(phi_old.vector(), t)
    timeseries_p.store(p_old.vector(), t)
    timeseries_v.store(v_old.vector(), t)
    timeseries_edge.store(edge_old.vector(), t)
    timeseries_bulk.store(bulk_region_old.vector(), t)

    # molecular field evolution
    if t < 1 or t > 4:
        field = Expression(('0.0','0.0'), degree=2)
        fieldmag = 0
    else:
        field = Expression(('1.0','0.0'), degree=2)
        fieldmag =1

    #VELOCITY
    L_v = zeta*inner(outer(p_old, p_old), nabla_grad(y)) * dx
    solve(a_v == L_v, vpr_new, bcs_flow, solver_parameters=dict(linear_solver='superlu_dist',
                                                                      preconditioner='ilu'))
    # POLARITY EVOLUTION #
    L_pol = (1. / dt) * dot(p_old, yp) * dx - inner(nabla_grad(p_old) * (v_new + w_sa * p_old), yp) * dx - \
            (alpha / phicr) * inner((phi_old - phicr) * p_old, zp) * dx + \
            dot(p_old, p_old) * alpha * inner(p_old, zp) * dx - \
            cE * (1 + delta_ph * edge_old) * inner(field, zp) * dx + \
            beta * inner(nabla_grad(phi_old), zp) * dx

    solve(a_pol == L_pol, pols_new, bcs_pol, solver_parameters=dict(linear_solver='superlu_dist',
                                                                 preconditioner='ilu'))

    # PHASE FIELD PROBLEM#
    L_phi = (1. / dt) * phi_old * w1 * dx - div(phi_old * (v_new + w_sa * p_old)) * w1 * dx + \
            (a / (2 * phicr ** 4)) * phi_old * (phi_old - phi0) * (2 * phi_old - phi0) * w2 * dx - \
            (alpha / (2 * phicr)) * dot(p_old, p_old) * w2 * dx - \
            beta * div(p_new) * w2 * dx
    solve(a_phi == L_phi, phis_new, bcs_phi, solver_parameters=dict(linear_solver='superlu_dist',
                                                                      preconditioner='ilu'))

    #Move regions
    L_lead = (1. / dt) * leading_edge_old * w_region * dx - div(
        leading_edge_old * (v_new + w_sa * p_old)) * w_region * dx
    solve(a_region == L_lead, leading_edge, bcs_region, solver_parameters=dict(linear_solver='superlu_dist',
                                                                               preconditioner='ilu'))

    L_trail = (1. / dt) * trailing_edge_old * w_region * dx - div(
        trailing_edge_old * (v_new + w_sa * p_old)) * w_region * dx
    solve(a_region == L_trail, trailing_edge, bcs_region, solver_parameters=dict(linear_solver='superlu_dist',
                                                                                 preconditioner='ilu'))

    L_top = (1. / dt) * top_edge_old * w_region * dx - div(top_edge_old * (v_new + w_sa * p_old)) * w_region * dx
    solve(a_region == L_top, top_edge, bcs_region, solver_parameters=dict(linear_solver='superlu_dist',
                                                                          preconditioner='ilu'))

    L_bottom = (1. / dt) * bottom_edge_old * w_region * dx - div(
        bottom_edge_old * (v_new + w_sa * p_old)) * w_region * dx
    solve(a_region == L_bottom, bottom_edge, bcs_region, solver_parameters=dict(linear_solver='superlu_dist',
                                                                                preconditioner='ilu'))
    L_bulk = (1. / dt) * bulk_region_old * w_region * dx - div(bulk_region_old * (v_new + w_sa * p_old)) * w_region * dx
    solve(a_region == L_top, bulk_region, bcs_region, solver_parameters=dict(linear_solver='superlu_dist',
                                                                             preconditioner='ilu'))

    L_edge = (1. / dt) * edge_old * w_region * dx - div(edge_old * (v_new + w_sa * p_old)) * w_region * dx
    solve(a_region == L_edge, edge_region, bcs_region, solver_parameters=dict(linear_solver='superlu_dist',
                                                                              preconditioner='ilu'))

    # ASSIGN ALL VARIABLES FOR NEW STEP
    polarity_assigner_inv.assign(p_old, pols_new.sub(0))
    velocity_assigner_inv.assign(v_old, vpr_new.sub(0))
    phi_assigner_inv.assign(phi_old,phis_new.sub(0))

    # Compute leading edge outgrowth
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: leading_edge_old(x) > 0)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        sumstat[i, 0] = U * assemble(v_old[0] * dx_sub(1)) / area
        sumstat[i, 1] = 100 * w_sa * assemble((1/(1+f_field*fieldmag))*(1+f_field*dot(p_old,field)) * p_old[0] * dx_sub(1)) / area
    except Exception as e:
        print('leading', i, e)

    # Compute trailing edge outgrowth
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: trailing_edge_old(x) >0)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        sumstat[i, 2] = U * assemble(v_old[0] * dx_sub(1)) / area
        sumstat[i, 3] = 100 * w_sa * assemble((1/(1+f_field*fieldmag))*(1+f_field*dot(p_old,field)) * p_old[0] * dx_sub(1)) / area
    except Exception as e:
        print('trailing', i, e)

    # Compute top zone speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: top_edge_old(x) > 0)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        top_vel = U * assemble(v_old[0] * dx_sub(1)) / area
        top_pol = 100 * w_sa * assemble((1/(1+f_field*fieldmag))*(1+f_field*dot(p_old,field)) * p_old[0] * dx_sub(1)) / area
    except Exception as e:
        print('top', i, e)

    # Compute bottom zone speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: bottom_edge_old(x) >0)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        bottom_vel = U * assemble(v_old[0] * dx_sub(1)) / area
        bottom_pol = 100 * w_sa * assemble((1/(1+f_field*fieldmag))*(1+f_field*dot(p_old,field)) * p_old[0] * dx_sub(1)) / area
    except Exception as e:
        print('bottom', i, e)

    try:
        sumstat[i, 4] = 0.5 * (top_vel + bottom_vel)
        sumstat[i, 5] = 0.5 * (top_pol + bottom_pol)
    except:
        pass

    # Compute bulk directionality and speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: bulk_region_old(x) > 0)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        sumstat[i, 6] = assemble((inner((1/(1+f_field*fieldmag))*(1+dot(p_old,field)) * p_old + v_old, E) / sqrt(inner((1/(1+f_field*fieldmag))*(1+dot(p_old,field)) * p_old + v_old, (1/(1+f_field*fieldmag))*(1+dot(p_old,field)) * p_old + v_old))) * dx_sub(1)) / area
        sumstat[i, 7] = U * assemble(v_old[0] * dx_sub(1)) / area
        sumstat[i, 8] = 100 * w_sa * assemble((1/(1+f_field*fieldmag))*(1+dot(p_old,field)) * p_old[0] * dx_sub(1)) / area
        sumstat[i, 9] = assemble(abs(100*(1/(1+f_field*fieldmag))*(1+f_field*dot(p_old,field)) * p_old[0] + v_old[0]) * dx_sub(1)) / area

    except Exception as e:
        print('bulk', i, e)

np.savetxt('linear/' + 'track_interface_1_-5' + '.txt', sumstat)
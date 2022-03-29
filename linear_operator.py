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

# Set fenics parameters
parameters["form_compiler"]["quadrature_degree"] = 3

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
delta_ph = float(sys.argv[1])
set_log_level(20)

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



self.boundary = 'near(x[1],20) || near(x[1], 80) || near(x[0], 0) || near(x[0],60)'
self.n = FacetNormal(mesh)

# Assign initial conditions for velocity and pressure
self.v_old = interpolate(vIC(), V)
self.pr_old = TestFunction(W)  # interpolate(scalarZero(),W)

# Assign initial conditions for phi fields
self.phi_old = interpolate(phiIC(), W)
self.phider_old = TestFunction(W)  # interpolate(scalarZero(),W)

# Assign initial conditions for polarity fields
self.p_old = interpolate(pIC(), V)
self.pder_old = TestFunction(V)  # interpolate(vIC(),V)

    def E(self, u):
        return sym(nabla_grad(u))

    def W(self, u):
        return skew(nabla_grad(u))

#Define the variational test functions

phis_new = Function(phasespace)







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





#POLARITY PROBLEM
pols_new = Function(polarityspace)
p_new, pder_new = split(pols_new)
yz = TestFunction(polarityspace)
yp, zp = split(yz)

dP = TrialFunction(polarityspace)
(dp1, dp2) = split(dP)

a_pol = (1. / dt) * dot(dp1,yp) * dx + (1. / Gamma) * inner(dp2, yp) * dx + \
       inner(dp2, zp) * dx - kappa * inner(nabla_grad(dp1), nabla_grad(zp)) * dx \


L_pol = (1. / dt) * dot(p_old, yp) * dx - inner(nabla_grad(p_old) * (v_new + w_sa * p_old), yp) * dx - \
       (alpha / phicr) * inner((phi_old - phicr) * p_old, zp) * dx + \
       dot(p_old, p_old) * alpha * inner(p_old, zp) * dx - \
       cE * (1 + delta_ph*inner(nabla_grad(phi_old),nabla_grad(phi_old))/(1+inner(nabla_grad(phi_old),nabla_grad(phi_old)))) * inner(field, zp) * dx + \
       beta * inner(nabla_grad(phi_old), zp) * dx

# PHASE FIELD PROBLEM#
phi_new, phider_new = split(phis_new)
w1w2 = TestFunction(phasespace)
w1, w2 = split(w1w2)
dphi = TrialFunction(phasespace)
(dphi1, dphi2) = split(dphi)

# phi evolution


a_phi = (1. / dt) * dphi1 * w1 * dx + M * dot(nabla_grad(dphi2), nabla_grad(w1)) * dx +\
        dphi2 * w2 * dx - k * dot(nabla_grad(dphi1), nabla_grad(w2)) * dx \

L_phi = (1. / dt) * phi_old * w1 * dx - div(phi_old * (v_new + w_sa * p_old)) * w1 * dx +\
        (a / (2 * phicr ** 4)) * phi_old * (phi_old - phi0) * (2 * phi_old - phi0) * w2 * dx -\
        (alpha / (2 * phicr)) * dot(p_old, p_old) * w2 * dx -\
        beta * div(p_new) * w2 * dx



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


    # POLARITY EVOLUTION #
    L_pol = (1. / dt) * dot(p_old, y) * dx - dot(nabla_grad(p_old) * (v_old + w_sa * p_old), y) * dx
    solve(a_pol == L_pol, p_new, bcs_pol, solver_parameters=dict(linear_solver='gmres',
                                                                 preconditioner='ilu'))
    print('polarity', p_new.vector().get_local().min(), p_new.vector().get_local().max())

    # STRESS TENSOR
    L_str = eta * inner(sym(nabla_grad(v_old)), z) * dx + (eta / E_bulk * dt) * inner(str_old, z) * dx
    solve(a_str == L_str, str_new, bcs=bcs_str, solver_parameters=dict(linear_solver='gmres',
                                                                 preconditioner='ilu'))
    print('stress', str_new.vector().get_local().min(), str_new.vector().get_local().max())

    # FLOW PROBLEM#
    L_flow = - zeta * dot(div(outer(p_new, p_new)), y1) * dx
    solve(a_flow == L_flow, vpr_new, bcs_flow, solver_parameters=dict(linear_solver='gmres',
                                                                 preconditioner='ilu'))
    print('velocity', vpr_new.sub(0).vector().get_local().min(), vpr_new.sub(0).vector().get_local().max())
    # PHASE FIELD PROBLEM#
    L_phi = (1. / dt) * phi_old * w2 * dx + dot(v_new, nabla_grad(phi_old)) * w2 * dx
    solve(a_phi == L_phi, phi_new, bcs_phi, solver_parameters=dict(linear_solver='gmres',
                                                                      preconditioner='ilu'))
    print('phi', phi_new.vector().get_local().min(), phi_new.vector().get_local().max())

    # ASSIGN ALL VARIABLES FOR NEW STEP
    str_old.assign(str_new)
    p_old.assign(p_new)
    velocity_assigner_inv.assign(v_old, vpr_new.sub(0))
    phi_old.assign(phi_new)
    pressure_assigner_inv.assign(pr_old, vpr_new.sub(1))














    # Rename variables
    phi = solver.phi_old
    p = solver.p_old
    v = solver.v_old

    # Advance one time step in the simulation
    solver.advance_one_step(t)

    # Compute gradients of phase field to ID regions
    phigrad = project(grad(phi), V)
    angle_hor = project(-inner(grad(phi), right)/sqrt(inner(grad(phi),grad(phi))+0.005), W)
    angle_ver = project(-inner(grad(phi), up)/sqrt(inner(grad(phi),grad(phi))+0.005), W)

    # Compute leading edge outgrowth
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: angle_hor(x) > min_angle)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        sumstat[i, 0] = U * assemble(v[0] * dx_sub(1)) / area
        sumstat[i, 1] = 100*w_sa * assemble(p[0] * dx_sub(1)) / area
    except Exception as e:
        print('leading', i, e)

    # Compute trailing edge outgrowth
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: angle_hor(x) < -min_angle)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        sumstat[i, 2] = U * assemble(v[0] * dx_sub(1)) / area
        sumstat[i, 3] = 100*w_sa * assemble(p[0] * dx_sub(1)) / area
    except Exception as e:
        print('trailng', i, e)

    # Compute top zone speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: angle_ver(x) > min_angle)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        top_vel = U * assemble(v[0] * dx_sub(1)) / area
        top_pol = 100*w_sa * assemble(p[0] * dx_sub(1)) / area
    except Exception as e:
        print('top', i, e)

    # Compute bottom zone speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: angle_ver(x) < -min_angle)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        bottom_vel = U * assemble(v[0] * dx_sub(1)) / area
        bottom_pol = 100*w_sa * assemble(p[0] * dx_sub(1)) / area
    except Exception as e:
        print('bottom', i, e)

    try:
        sumstat[i, 4] = 0.5 * (top_vel + bottom_vel)
        sumstat[i, 5] = 0.5 * (top_pol + bottom_pol)
    except:
        pass

    # Compute bulk directionality and speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: phi(x) > 0.5)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        sumstat[i, 6] = assemble((inner(100*p+v, E) / sqrt(inner(100*p+v, 100*p+v))) * dx_sub(1)) / area
        sumstat[i, 7] = U * assemble(v[0] * dx_sub(1)) / area
        sumstat[i, 8] = 100*w_sa * assemble(p[0] * dx_sub(1)) / area
        sumstat[i,9] = assemble(abs(100*p[0] + v[0]) * dx_sub(1)) / area

    except Exception as e:
        print('bulk', i, e)

np.savetxt('delta_ph_grad/'+'test_delta_ph_'+str(delta_ph).replace('.','_')+'.txt',sumstat)


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
w_sa = 100
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
num_w = 25
num_u = 25

# Set simulation parameters we do inference on
cE = 0.05
beta = 0.075
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


class NSSolver:
    def __init__(self):
        # Define the boundaries
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

    def advance_one_step(self, t):
        # Load objects from previous time step
        v_old = self.v_old
        pr_old = self.pr_old

        phi_old = self.phi_old
        phider_old = self.phider_old

        p_old = self.p_old
        pder_old = self.pder_old

        vpr_new = Function(flowspace)
        phis_new = Function(phasespace)
        pols_new = Function(polarityspace)

        # FLOW PROBLEM#
        yw = TestFunction(flowspace)
        y, w = split(yw)

        dU = TrialFunction(flowspace)
        (du1, du2) = split(dU)

        v_new, pr_new = split(vpr_new)

        # Navier-Stokes scheme
        print('Navier-Stokes')
        F_v = eta*inner(nabla_grad(v_new), nabla_grad(y)) * dx + \
              gamma * inner(v_new, y) * dx + dot(nabla_grad(pr_new), y) * dx - \
              inner(outer(p_old, p_old), nabla_grad(y)) * dx

        F_incomp = div(v_new) * w * dx  # corresponding to incompressibility condition
        F_flow = F_v + F_incomp  # total variational formulation of flow problem

        # Set boundary conditions#
        zero = Expression(('0.0', '0.0', '0.0'), degree=2)  # Expression(('0.0','0.0','0.0'), degree=2)
        bcs = DirichletBC(flowspace, zero, self.boundary)  # set zero boundary condition

        J = derivative(F_flow, vpr_new, dU)
        solve(F_flow == 0, vpr_new, bcs=bcs, J=J)  # solve the nonlinear variational problem
        v_new, pr_new = split(vpr_new)
        # POLARITY PROBLEM#
        p_new, pder_new = split(pols_new)
        yz = TestFunction(polarityspace)
        y, z = split(yz)

        dU = TrialFunction(polarityspace)
        (du1, du2) = split(dU)

        # polarity evolution
        print('Polarity')
        F_p = (1. / dt) * dot(p_new - p_old, y) * dx + inner(nabla_grad(p_new) * (v_new + w_sa * p_new), y) * dx + \
              (1. / Gamma) * inner(pder_new, y) * dx

        # molecular field evolution
        if t < 1 or t > 4:
            field = interpolate(vIC(), V)
        else:
            field = interpolate(pointRight(), V)

        # F_pder = inner(pder_new, z) * dx + (alpha / phicr) * inner((phi_old - phicr) * p_new, z) * dx - \
        #          dot(p_old, p_old) * alpha * inner(p_new, z) * dx - \
        #          cE * inner(field, field) * inner(p_new - field, p_new - field) * inner(p_new - field, z) * dx - \
        #          kappa * inner(nabla_grad(p_new), nabla_grad(z)) * dx - beta * inner(nabla_grad(phi_old), z) * dx

        F_pder = inner(pder_new, z) * dx + (alpha / phicr) * inner((phi_old - phicr) * p_new, z) * dx - \
               dot(p_old, p_old) * alpha * inner(p_new, z) * dx + \
               cE * inner(field, z) * dx - \
               kappa * inner(nabla_grad(p_new), nabla_grad(z)) * dx - beta * inner(nabla_grad(phi_old), z) * dx

        F_pols = F_p + F_pder
        J = derivative(F_pols, pols_new, dU)

        # Set boundary conditions#
        zero = Expression(('0.0', '0.0', '0.0', '0.0'), degree=2)
        bcs = DirichletBC(polarityspace, zero, self.boundary)  # set zero boundary condition
        solve(F_pols == 0, pols_new, J=J, bcs=bcs)  # solve the nonlinear variational problem
        p_new, pder_new = split(pols_new)

        # PHASE FIELD PROBLEM#
        print('Phi')
        phi_new, phider_new = split(phis_new)
        w1w2 = TestFunction(phasespace)
        w1, w2 = split(w1w2)
        dU = TrialFunction(phasespace)
        (du1, du2) = split(dU)

        # phi evolution
        F_phi = (1. / dt) * (phi_new - phi_old) * w1 * dx + div(phi_new * (v_new + w_sa * p_new)) * w1 * dx + \
                M * dot(nabla_grad(phider_new), nabla_grad(w1)) * dx

        F_phider = phider_new * w2 * dx - (a / (2 * phicr ** 4)) * phi_new * (phi_new - phi0) * (
                    2 * phi_new - phi0) * w2 * dx - \
                   k * dot(nabla_grad(phi_new), nabla_grad(w2)) * dx + \
                   (alpha / (2 * phicr)) * dot(p_new, p_new) * w2 * dx + beta * div(p_new) * w2 * dx

        F_phase = F_phi + F_phider

        zero = Expression(('0.0', '0.0'), degree=2)
        bcs = DirichletBC(phasespace, zero, self.boundary)  # set zero boundary condition
        J = derivative(F_phase, phis_new, dU)

        solve(F_phase == 0, phis_new, bcs=bcs, J=J)
        phi_new, phider_new = split(phis_new)

        # ASSIGN ALL VARIABLES FOR NEW STEP
        # Flow problem variables
        v_new, pr_new = vpr_new.split(True)
        self.v_old.assign(v_new)

        try:
            self.pr_old.assign(pr_new)
        except:
            self.pr_old = pr_new

        # Polarity problem variables
        p_new, pder_new = pols_new.split(True)
        self.p_old.assign(p_new)

        try:
            self.pder_old = pder_new
        except:
            self.pder_old.assign(pder_new)

        # Phase problem variables
        phi_new, phider_new = phis_new.split(True)
        self.phi_old.assign(phi_new)

        try:
            self.phider_old = phider_new
        except:
            self.phider_old.assign(phider_new)


# Defining the problem
solver = NSSolver()

# Make files to store the simulation data
timeseries_phi = TimeSeries('results/phi_bestfit')
timeseries_p =  TimeSeries('results/p_bestfit')
timeseries_v = TimeSeries('results/v_bestfit')

for i in tqdm(range(numSteps)):
    t = i * dt
    # Rename variables
    phi = solver.phi_old
    p = solver.p_old
    v = solver.v_old

    # Save variables
    timeseries_phi.store(phi.vector(), t)
    timeseries_p.store(p.vector(), t)
    timeseries_v.store(v.vector(), t)

    # Advance one time step in the simulation
    solver.advance_one_step(t)


from dolfin import *
import math
from fenics import *
from scipy.integrate import odeint
import numpy as np
from ufl import nabla_div
from tqdm import tqdm
import sys

# Simulator settings
num_points = 50
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
w_sa = 0.0015 * 3600
phi0 = 2
phicr = 1
D = 0.007
M = D / a
gamma = 0.04
zeta = 0.01
a = 1
beta = cE=1
Gamma = 1. / 5
dt = 0.01
min_angle = 0.05
minphi = 0.5
minphi_b = 0.25
numSteps = int(10 / dt)  # electrotaxis time is 10 hours
num_w = 25
num_u = 25

#Set fenics parameters
set_log_level(20)
parameters["form_compiler"]["quadrature_degree"] = 3
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"]=True
parameters["krylov_solver"]["nonzero_initial_guess"] = True

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

        self.velocity_assigner = FunctionAssigner(flowspace.sub(0), V)
        self.pressure_assigner = FunctionAssigner(flowspace.sub(1), W)
        self.polarity_assigner = FunctionAssigner(flowspace.sub(0), V)
        self.pder_assigner = FunctionAssigner(flowspace.sub(1), W)

        yw = TestFunction(flowspace)
        y, w = split(yw)
        dU = TrialFunction(flowspace)
        (du1, du2) = split(dU)
        a = inner(nabla_grad(du1), nabla_grad(y)) * dx + \
            gamma * inner(du1, y) * dx + dot(nabla_grad(du2), y) * dx
        self.A_flow = assemble(a)

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
        v_new, pr_new = split(vpr_new)
        self.velocity_assigner.assign(vpr_new.sub(0), v_old)
        self.pressure_assigner.assign(vpr_new.sub(1), pr_old)
        L = inner(phi_old * outer(p_old, p_old), nabla_grad(y)) * dx
        b = assemble(L)
        solver = KrylovSolver("gmres", "ilu")
        solver.set_operator(self.A_flow)
        solver.solve(vpr_new.vector(), b)
        v_new, pr_new = split(vpr_new)

        # POLARITY PROBLEM#
        p_new, pder_new = split(pols_new)
        yz = TestFunction(polarityspace)
        y, z = split(yz)
        dU = TrialFunction(polarityspace)
        (du1, du2) = split(dU)

        # polarity evolution
        if t < 1 or t > 4:
            field = interpolate(vIC(), V)
        else:
            field = interpolate(pointRight(), V)

        a = (1. / dt) * dot(du1, y) * dx + inner(nabla_grad(du1) * (v_new + w_sa * du1), y) * dx +\
              (1. / Gamma) * inner(du2, y) * dx + inner(du2, z) * dx +\
              (alpha / phicr) * inner((phi_old - phicr) * du1, z) * dx -\
              dot(p_old, p_old) * alpha * inner(du1, z) * dx -\
              kappa * inner(nabla_grad(p_new), nabla_grad(z)) * dx
        L = (1. / dt) * dot(p_old, y) * dx + beta * inner(nabla_grad(phi_old), z) * dx + \
             cE * inner(field, field) * inner(p_old - field, p_old - field) * inner(p_old - field, z) * dx
        A = assemble(a)
        b = assemble(L)
        self.polarity_assigner.assign(pols_new.sub(0), p_old)
        self.pder_assigner.assign(pols_new.sub(1), pder_old)
        solver = KrylovSolver("gmres", "ilu")
        solver.set_operator(A)
        solver.solve(pols_new.vector(), b)
        p_new, pder_new = split(pols_new)

        # PHASE FIELD PROBLEM#
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

for i in tqdm(range(numSteps)):
    t = i * dt
    solver.advance_one_step(t)
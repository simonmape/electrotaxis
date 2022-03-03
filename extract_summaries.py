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
w_sa = 50
phi0 = 2
phicr = 1
D = 0.007
M = D / a
gamma = 0.04
zeta = 0.01
a = 1
# cE=1
Gamma = 1. / 5
dt = 0.01
min_angle = 0.05
minphi = 0.5
minphi_b = 0.25
numSteps = int(10 / dt)  # electrotaxis time is 10 hours
num_w = 25
num_u = 25
U = 1
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


betas = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05,
        1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05])
cEs = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05,
        1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05])
Gamma = 0.2

for beta in tqdm(betas):
    for cE in cEs:
        summary = np.zeros((numSteps, 8))
        timeseries_phi = TimeSeries(
            'results/phi_G' + str(Gamma).replace('.', '_') + '_C' + str(cE).replace('.', '_') + '_b' + str(beta).replace('.',
                                                                                                                         '_'))
        timeseries_p = TimeSeries(
            'results/p_G' + str(Gamma).replace('.', '_') + '_C' + str(cE).replace('.', '_') + '_b' + str(beta).replace('.',
                                                                                                                       '_'))
        timeseries_v = TimeSeries(
            'results/v_G' + str(Gamma).replace('.', '_') + '_C' + str(cE).replace('.', '_') + '_b' + str(beta).replace('.',
                                                                                                                       '_'))

        # Define the functions to be loaded here
        scalar_space = FunctionSpace(mesh, P1)
        E = interpolate(fixedField(), V)
        right = interpolate(pointRight(), V)
        up = interpolate(pointUp(), V)
        phi_load = Function(scalar_space)
        v_load = Function(V)
        p_load = Function(V)


        for i in range(numSteps):
            t = i * dt

            # Retrieve values of variables at time t
            timeseries_phi.retrieve(phi_load.vector(), t)
            timeseries_v.retrieve(v_load.vector(), t)
            timeseries_p.retrieve(p_load.vector(), t)

            # Compute gradients of phase field to ID regions
            phigrad = project(grad(phi_load), V)
            angle_hor = project(-inner(grad(phi_load), right), W)  # /sqrt(inner(phigrad,phigrad)),W)
            angle_ver = project(-inner(grad(phi_load), up, ), W)  # /sqrt(inner(phigrad,phigrad)),W)

            # Compute leading edge outgrowth
            cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
            region = AutoSubDomain(lambda x, on: angle_hor(x) > min_angle)
            region.mark(cf, 1)
            dx_sub = Measure('dx', subdomain_data=cf)
            area = assemble(E[0] * dx_sub(1))
            try:
                summary[i, 0] = assemble((U * v_load[0] + w_sa * p_load[0]) * dx_sub(1)) / area
            except Exception as e:
                print(i, e)

            # Compute trailing edge outgrowth
            cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
            region = AutoSubDomain(lambda x, on: angle_hor(x) < -min_angle)
            region.mark(cf, 1)
            dx_sub = Measure('dx', subdomain_data=cf)
            area = assemble(E[0] * dx_sub(1))
            try:
                summary[i, 1] = assemble((U * v_load[0] + w_sa * p_load[0]) * dx_sub(1)) / area
            except Exception as e:
                print(i, e)

            # Compute top zone speed
            cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
            region = AutoSubDomain(lambda x, on: angle_ver(x) > min_angle)
            region.mark(cf, 1)
            dx_sub = Measure('dx', subdomain_data=cf)
            area = assemble(E[0] * dx_sub(1))
            try:
                summary[i, 2] = assemble((U * v_load[0] + w_sa * p_load[0]) * dx_sub(1)) / area
            except Exception as e:
                print(i, e)

            # Compute bottom zone speed
            cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
            region = AutoSubDomain(lambda x, on: angle_ver(x) < -min_angle)
            region.mark(cf, 1)
            dx_sub = Measure('dx', subdomain_data=cf)
            area = assemble(E[0] * dx_sub(1))
            try:
                summary[i, 3] = assemble((U * v_load[0] + w_sa * p_load[0]) * dx_sub(1)) / area
            except Exception as e:
                print(i, e)
            # Compute bulk directionality and speed
            cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
            region = AutoSubDomain(lambda x, on: phi_load(x) >= 0.2 and abs(angle_hor(x)) + abs(angle_ver(x)) < 0.3)
            region.mark(cf, 1)
            dx_sub = Measure('dx', subdomain_data=cf)
            area = assemble(E[0] * dx_sub(1))
            try:
                summary[i, 4] = assemble((inner((U * v_load + w_sa * p_load), E) / sqrt(
                    inner((U * v_load + w_sa * p_load), (U * v_load + w_sa * p_load)))) * dx_sub(1)) / area
                summary[i, 5] = assemble((U * v_load[0] + w_sa * p_load[0]) * dx_sub(1)) / area
                summary[i, 6] = assemble((U * v_load[0]) * dx_sub(1)) / area
                summary[i, 7] = assemble((w_sa * p_load[0]) * dx_sub(1)) / area
            except Exception as e:
                print(i, e)
        # Save output
        fname = 'sumstats2/G' + str(Gamma).replace('.', '_') + '_C' + str(cE).replace('.', '_') + '_b' + str(
            beta).replace('.', '_')  + '.txt'
        np.savetxt(fname, summary)


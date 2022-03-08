from dolfin import *
import math
from fenics import *
from scipy.integrate import odeint
import numpy as np
from ufl import nabla_div
from tqdm import tqdm
import sys
import time

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
dt = 0.01
min_angle = 0.01
minphi = 0.5
minphi_b = 0.25
numSteps = int(10 / dt)  # electrotaxis time is 10 hours
set_log_level(20)

cE = float(sys.argv[1])
beta = float(sys.argv[2])
Gamma = 0.2
U_list = [1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000,
          4200, 4400, 4600, 4800, 5000, 5200, 5400]
w_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

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

#summary[:,0]= leading zone speed
#summary[:,1] = leading zone polarity
#summary[:,2] = trailing zone speed
#summary[:,3] = trailing zone polarity
#summary[:,4] = top/bottom zone speed
#summary[:,5] = top/bottom zone polarity
#summary[:,6] = bulk directionality
#summary[:,7] = bulk speed
#summary[:,8] = bulk polarity


summary = np.zeros((numSteps, 9))
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
bulk_region = []

for i in range(numSteps):
    t = i * dt
    # Retrieve values of variables at time t
    timeseries_phi.retrieve(phi_load.vector(), t)
    timeseries_v.retrieve(v_load.vector(), t)
    timeseries_p.retrieve(p_load.vector(), t)

    # Compute gradients of phase field to ID regions
    phigrad = project(grad(phi_load), V)
    angle_hor = project(-inner(grad(phi_load), right), W)
    angle_ver = project(-inner(grad(phi_load), up, ), W)

    # Compute leading edge outgrowth
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: angle_hor(x) > min_angle)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        summary[i, 0] = assemble(v_load[0] * dx_sub(1)) / area
        summary[i, 1] = assemble(p_load[0] * dx_sub(1)) / area
    except Exception as e:
        print(i, e)

    # Compute trailing edge outgrowth
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: angle_hor(x) < -min_angle)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        summary[i, 2] = assemble(v_load[0]  * dx_sub(1)) / area
        summary[i, 3] = assemble(p_load[0] * dx_sub(1)) / area
    except Exception as e:
        print(i, e)

    # Compute top zone speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: angle_ver(x) > min_angle)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        top_vel = assemble(v_load[0] * dx_sub(1)) / area
        top_pol = assemble(p_load[0] * dx_sub(1)) / area
    except Exception as e:
        print(i, e)

    # Compute bottom zone speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: angle_ver(x) < -min_angle)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    area = assemble(E[0] * dx_sub(1))
    try:
        bottom_vel = assemble(v_load[0] * dx_sub(1)) / area
        bottom_pol = assemble(p_load[0] * dx_sub(1)) / area
    except Exception as e:
        print(i, e)

    try:
        summary[i,4] = 0.5 * (top_vel + bottom_vel)
        summary[i,5] = 0.5 * (top_pol + bottom_pol)
    except:
        pass

    # Compute bulk directionality and speed
    cf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    region = AutoSubDomain(lambda x, on: phi_load(x) >= 0.2 and abs(angle_hor(x)) + abs(angle_ver(x)) < 0.3)
    region.mark(cf, 1)
    dx_sub = Measure('dx', subdomain_data=cf)
    bulk_region.append(dx_sub)
    area = assemble(E[0] * dx_sub(1))
    try:
        summary[i, 7] = assemble(v_load[0] * dx_sub(1)) / area
        summary[i, 8] = assemble(p_load[0] * dx_sub(1)) / area
    except Exception as e:
        print(i, e)

for U in U_list:
    for w_sa in w_list:
        sumstat = np.zeros((numSteps, 9))
        sumstat[:, 0] = U * summary[:, 0]
        sumstat[:, 1] = w_sa * summary[:, 1]
        sumstat[:, 2] = U * summary[:, 2]
        sumstat[:, 3] = w_sa * summary[:, 3]
        sumstat[:, 4] = U * summary[:, 4]
        sumstat[:, 5] = w_sa * summary[:, 5]
        sumstat[:, 7] = U * summary[:, 7]
        sumstat[:, 8] = w_sa * summary[:, 8]

        for i in range(numSteps):
            t = i * dt
            # Retrieve values of variables at time t
            timeseries_phi.retrieve(phi_load.vector(), t)
            timeseries_v.retrieve(v_load.vector(), t)
            timeseries_p.retrieve(p_load.vector(), t)

            dx_sub = bulk_region[i]
            area = assemble(E[0] * dx_sub(1))
            try:
                sumstat[i, 6] = assemble((inner((U * v_load + w_sa * p_load), E) / sqrt(
                    inner((U * v_load + w_sa * p_load), (U * v_load + w_sa * p_load)))) * dx_sub(1)) / area
            except Exception as e:
                print(i, e)

        # Save output
        fname = 'sumstats/G' + str(Gamma).replace('.', '_') + '_C' + str(cE).replace('.', '_') + '_b' + str(
            beta).replace('.', '_') + '_w' + str(w_sa).replace('.', '_') + '_u' + str(U).replace('.', '_') + '.txt'
        print(fname)
        np.savetxt(fname, sumstat)
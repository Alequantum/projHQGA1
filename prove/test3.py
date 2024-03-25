import problems as p, hqga_utils, utils, hqga_algorithm
from utils import computeHammingDistance

from qiskit import Aer
import math

from qiskit import execute
import copy
from tqdm import tqdm
from qiskit.providers.jobstatus import JOB_FINAL_STATES
import itertools
import hqga_utils
from qiskit_experiments.library import StateTomography
from qiskit.quantum_info import state_fidelity
import random
from qiskit.quantum_info import  DensityMatrix

import math
import random
import itertools

import discretization as dis

simulator = Aer.get_backend('qasm_simulator')
device_features= hqga_utils.device(simulator, False)

params= hqga_utils.Parameters(base=3,pop_size=3, max_gen=5, prob_mut=0.3,elitism="Quantistic_elitism")
params.draw_circuit=True

problem = p.SphereProblem(num_bit_code=1,base=params.base)

circuit = hqga_utils.setupCircuit(params.pop_size, problem.dim * problem.num_bit_code)


creal=dis.convertFromBinToFloat('11',lower_bounds = [-5.12,-5.12], upper_bounds = [5.12,5.12] , num_bit_code=1, dim=2, base=3)
print(creal)
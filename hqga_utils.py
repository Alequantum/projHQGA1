from qiskit import(QuantumCircuit, ClassicalRegister, QuantumRegister)
import copy
import random
import math
import numpy as np


ELITISM_R= "Reinforcement_elitism"
ELITISM_Q= "Quantistic_elitism"
ELITISM_D ="Deterministic_elitism"


def generate_input_strings(num_variabili, p):
    """
    Genera tutte le possibili stringhe di input basate sul numero di variabili.

    Args:
        num_variabili (int): Numero di variabili, con ciascuna variabile rappresentata da due cifre.
        p (int): Numero di valori possibili per ogni coppia di cifre (0 a p-1).

    Returns:
        list: Lista di tutte le possibili stringhe di input.
    """
    input_strings = []

    # Genera tutte le combinazioni possibili
    def genera_combinazioni(prefix="", depth=num_variabili):
        if depth == 0:
            input_strings.append(prefix)
            return
        for i in range(p):
            #genera_combinazioni(prefix + f"{i:02}", depth - 1)#stringa doppia
            genera_combinazioni(prefix + f"{i}", depth - 1)#stringa singola
    genera_combinazioni()

    return input_strings

def stato_su_sfera_di_bloch(theta, phi):
    """ Crea un circuito quantistico che rappresenta uno stato sulla sfera di Bloch. """
    qc=QuantumCircuit(1)
    qc.ry(theta,0)
    qc.rz(phi,0)
    return qc
def distribute_points_on_sphere(n, iterations=1000, learning_rate=0.01):
    # Inizializza i punti in posizioni casuali sulla sfera
    points = np.random.randn(n, 3)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    for _ in range(iterations):
        # Calcola le forze di repulsione tra tutti i punti
        forces = np.zeros((n, 3))
        for i in range(n):
            for j in range(i + 1, n):
                # Calcola la differenza vettoriale e la distanza
                difference = points[i] - points[j]
                distance = np.linalg.norm(difference)
                # Aggiungi la forza di repulsione
                force = difference / (distance**3)
                forces[i] += force
                forces[j] -= force

        # Aggiorna i punti lungo la direzione della forza
        points += learning_rate * forces
        # Normalizza i punti per mantenerli sulla superficie della sfera
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    # Converti in coordinate sferiche (theta, phi)
    theta = np.arccos(points[:, 2]) # theta è l'angolo dalla direzione z
    phi = np.arctan2(points[:, 1], points[:, 0]) # phi è l'angolo dalla direzione x

    return np.vstack((theta, phi)).T

def prod(comb):
    """Calcola il prodotto dei valori in una combinazione."""
    risultato = 1
    for valore in comb:
        risultato *= valore
    return risultato


def isBetter(fit_a, fit_b, optMax):
    """Function that returns True if fit_a > fit_b if the problem is to be maximized (False otherwise), and
    if fit_a < fit_b if the problem is to be minimized (False otherwise)"""
    if optMax:
        if fit_a > fit_b:
            return True
        return False
    else:
        if fit_a < fit_b:
            return True
        return False


class globalBest:
    """Class that defines information about the best solution found during the evolution"""
    def __init__(self):
        self.chr = []
        self.phenotype = []
        self.fitness = None
        self.gen = None

    def display(self):
        print("The best chromosome is: ", self.chr)
        print("The best phenotype is: ", self.phenotype)
        print("Its fitness value is: ", self.fitness)
        print("The fitness evaluations to obtain the best: ", self.gen)

def fromQtoC(max_counts):
    """Function that returns the classical chromosomes starting from the outcome of the quantum circuit"""
    max_counts = ''.join(reversed(max_counts))
    classical_chromosomes=max_counts.split(" ")
    return classical_chromosomes


def computeFitnesses(classical_chromosomes, fitness_f):
    """Function that returns the fitness values of the classical chromosomes"""
    fitnesses=[]
    for chr in classical_chromosomes:
        fitnesses.append(fitness_f(chr))
    return fitnesses

def computeBest(fitnesses, optMax):
    """Fitness that computes the best solution starting form the fitness values and the kind of problem"""
    if optMax:
        best_fitness=max(fitnesses)
    else:
        best_fitness = min(fitnesses)
    return best_fitness, fitnesses.index(best_fitness)

def setupCircuit(number_of_populations, gene_per_chromosome):
    """Create a Quantum Circuit composed of the opportune number of qubits"""

    circuit = QuantumCircuit()

    for i in range(number_of_populations):
        circuit.add_register(QuantumRegister(gene_per_chromosome))

    for i in range(number_of_populations):
        circuit.add_register(ClassicalRegister(gene_per_chromosome))

    return circuit

def resetCircuit(circuit):
    """Remove all gates from the quantum circuit used during evolution"""
    list_obj=copy.deepcopy(circuit.data)
    for obj in list_obj:
        circuit.data.remove(obj)

def getMaxProbKey(counts):
    """Function that returns the classical chromosome characterized by the maximum probability to be measured"""
    return max(counts, key = lambda k: counts[k])

def create_dict_chr(circuit, classical_chromosomes):
    """Function that returns a dictionary where the keys are the qubits and
    the values are the corresponsing classical states"""
    dict_chr={}
    chr="".join(classical_chromosomes)
    #print(chr)
    i=0
    for quantum_register in circuit.qregs:
        for qubit in quantum_register:
            dict_chr[qubit]=chr[i]
            i+=1
    return dict_chr




def initializeTheta(circuit): # modificato
    """Function that initializes the values of the angles used in the rotation gates"""
    dict={}
    for quantum_register in circuit.qregs:
        for qubit in quantum_register:
            theta=np.random.uniform(low=0.0, high=2*np.pi, size=3)
            dict[qubit] = theta
    return dict

'''
def updateThetaReinforcementWithinRange(dict_chr, theta, epsilon,index_best, num_genes):
    """Function that updates the angles of rotation gates during the reinforcement elitism"""
    i=0
    for key in dict_chr.keys():
        if i in range(index_best*num_genes,index_best*num_genes+num_genes):
            if dict_chr[key]=='1':
                theta[key]=theta[key]+epsilon
                if theta[key] > math.pi/2:
                    theta[key] =  math.pi/2
            else:
                theta[key] = theta[key] - epsilon
                if theta[key] < -math.pi/2:
                    theta[key] = - math.pi/2
        i+=1

def resetThetaReinforcement(dict_chr, theta, old_theta, old_index_best, num_genes):
    """function that resets the angles of the rotation gates during the reinforcement elitism"""
    i=0
    for key in dict_chr.keys():
        if i in range(old_index_best*num_genes,old_index_best*num_genes+num_genes):
            theta[key]=old_theta[key]
        i+=1
'''

def applyMutationOnListWithinRange(circuit, prob, list_mutation, theta):# modificato
    """Function that adds rotation gates in the quantum circuit to implement mutations"""
    for qubit in list_mutation:
        r= random.random()
        if r < prob:
            #print("I am here", r, prob)
            angoli_random = np.random.uniform(low=0.0, high=2 * np.pi, size=3)
            circuit.u(angoli_random[0],angoli_random[1],angoli_random[2], qubit)
            theta[qubit] = [a + b for a, b in zip(theta[qubit], angoli_random)]



def applyMultiRotationOnList(circuit, theta, list_qubit):#modificato
    """Functions that adds rotation gates in the quantum circuit to create the previous quantum state"""
    for qubit in list_qubit:
        circuit.u(theta[qubit][0],theta[qubit][1],theta[qubit][2], qubit)


def applyEntanglementOnList(circuit, index_best, list_entang, theta):
    """Function that adds cnot gates to implement crossover"""
    qr1 = circuit.qregs[index_best]
    i=0
    for q in list_entang:
        circuit.cnot(qr1[i], q)
        theta[q]=theta[qr1[i]]
        i+=1


def applyMeasureOperator(circuit):
    """Function that adds measurement gates to the quantum circuit"""
    for quantum_classical_registers in zip(circuit.qregs, circuit.cregs):
        for qubit_bit in zip(quantum_classical_registers[0], quantum_classical_registers[1]):
         circuit.measure(qubit_bit[0], qubit_bit[1])

def applyMultiHadamardOnList(circuit, list_qubit):
    """Function that adds hadamard gates to the quantum circuit"""
    for qubit in list_qubit:
        circuit.h(qubit)

def applyXOnList(circuit, list_qubit_X,dict_chr,qc):
    """Function that adds X gate to the quantum circuit to implement deterministic elitism"""
    for qubit in list_qubit_X:
        circuit.compose(qc[int(dict_chr[qubit])], qubits=qubit, inplace=True)

        #if dict_chr[qubit]=='1':
            #circuit.x(qubit)


def computeLists(circuit, index_best, number_of_populations,num_genes):
    """Function that computes different lists of qubits, each one will be characterized by the application
    of different gates"""
    list_pop = [i for i in range(number_of_populations)]
    list_pop.remove(index_best)
    random.shuffle(list_pop)
    point=math.ceil(num_genes/(number_of_populations-1))
    k=0
    qr1 = circuit.qregs[index_best]
    list_qubit_mutation=[]
    list_qubit_gate=[]
    list_qubit_gate.extend([q for q in qr1])
    list_qubit_entang=[]
    #print(qr1)
    for ind in list_pop:
        qr2=circuit.qregs[ind]
        for i in range(k,min(k+point, num_genes)):
            #print("[", k, ",", k+point-1, "]")
            list_qubit_entang.append(qr2[i])
        list_qubit_mutation.extend([qr2[e] for e in range(0,num_genes) if e not in range(k,min(k+point, num_genes))])
        k=k+point
    list_qubit_gate.extend(list_qubit_mutation)
    return list_qubit_gate, list_qubit_entang, list_qubit_mutation, []

def updateListXElitismD(circuit, index_best, list_qubit_gate, list_qubit_X):
    """Function that updates the list of qubits that are undergone to deterministic elitism"""
    for qb in circuit.qregs[index_best]:
        list_qubit_X.append(qb)
    for qb in list_qubit_X:
        list_qubit_gate.remove(qb)


def initializeLists(circuit):
    """Function that initializes the list of qubits that will be characterized by the application
    of different set of gates"""
    list_qubit_mutation=[]
    list_qubit_gate=[]
    list_qubit_entang=[]
    list_qubit_X = []
    for quantum_register in circuit.qregs:
        for qubit in quantum_register:
            list_qubit_gate.append(qubit)
    return list_qubit_gate, list_qubit_entang, list_qubit_mutation, list_qubit_X



class Parameters():
    """Class that defines the information related to the hyper-parameters of HQGA"""
    def __init__(self,base, pop_size, max_gen, prob_mut, elitism, num_shots=1, progressBar=False, verbose=True,draw_circuit=False, qobj_id=None):
        self.base = base
        self.pop_size=pop_size
        self.max_gen=max_gen
        self.prob_mut=prob_mut
        self.num_shots=num_shots
        self.elitism=elitism
        self.progressBar=progressBar
        self.verbose = verbose
        self.draw_circuit= draw_circuit
        self.qobj_id=qobj_id


    def __str__(self):
        return str(self.elitism)+ "_prob_mut_"+str(self.prob_mut)
'''
class ReinforcementParameters(Parameters):
    """Class that defines the information related to the hyper-parameters of HQGA in the case of the reinforcement elitism"""
    def __init__(self, pop_size,max_gen,epsilon_init, epsilon,prob_mut, elitism=ELITISM_R, num_shots=1, progressBar=False, verbose=True,draw_circuit=False,qobj_id=None):
        super().__init__(pop_size,max_gen,epsilon_init, prob_mut, elitism, num_shots, progressBar,verbose,draw_circuit,qobj_id)
        self.epsilon=epsilon

    def __str__(self):
        return str(self.elitism)+"_eps_init_"+str(self.epsilon_init)+ "_prob_mut_"+str(self.prob_mut)+"_eps_"+str(self.epsilon)
'''

class device():
    """Class that defines the information related to the device where HQGA will be run"""
    def __init__(self, device, isReal=False,noise_model=None, coupling_map=None, basis_gates=None):
        self.device=device
        self.noise_model=noise_model
        self.coupling_map= coupling_map
        self.basis_gates=basis_gates
        self.real=isReal

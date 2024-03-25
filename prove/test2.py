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

simulator = Aer.get_backend('qasm_simulator')
device_features= hqga_utils.device(simulator, False)

params= hqga_utils.Parameters(base=3,pop_size=3, max_gen=5, prob_mut=0.3,elitism="Quantistic_elitism")
params.draw_circuit=True

problem = p.SphereProblem(num_bit_code=2,base=params.base)

circuit = hqga_utils.setupCircuit(params.pop_size, problem.dim * problem.num_bit_code)









num_var=params.pop_size * problem.dim * problem.num_bit_code #numero di qubit del circuito
chromosome_evolution=[]
bests =[]
#gen=0
theta = hqga_utils.initializeTheta(circuit)
gBest = hqga_utils.globalBest()
list_qubit_gate, list_qubit_entang, list_qubit_mutation, list_qubit_X= hqga_utils.initializeLists(circuit)
dict_chr=[]
flag_index_best=False
gen=0
#while gen<=max_gen:


#print("\n########## generation #########", gen)
hqga_utils.applyMultiHadamardOnList(circuit, list_qubit_gate)
hqga_utils.applyMultiRotationOnList(circuit, theta, list_qubit_gate)
#hqga_utils.applyXOnList(circuit, list_qubit_X, dict_chr)
circuit.barrier()

#hqga_utils.applyMeasureOperator(circuit)
# Draw the circuit
if params.draw_circuit:
    print(circuit.draw(output="text", fold=300))
    print("Circuit depth is ",circuit.depth())

# Execute the circuit on the qasm simulator
while True:
    circuit.name =str(params.qobj_id)+str(0)
    try:
        if device_features.real:
            job = execute(circuit, device_features.device, shots=params.num_shots)#da modificare
        else:
            #job = execute(circuit, device_features.device, noise_model=device_features.noise_model,
                       #coupling_map=device_features.coupling_map,
                       #basis_gates=device_features.basis_gates, shots=params.num_shots)
            state_result = []
            for i in range(num_var):
                st = StateTomography(circuit, measurement_qubits=[i])
                stdata = st.run(device_features.device).block_for_results()
                state_result.append(stdata.analysis_results("state"))
        # Grab results from the job
        #result = job.result()
        #print(result)
        break
    except Exception as e:
        print(e)

distributed_points = hqga_utils.distribute_points_on_sphere(params.base).flatten().tolist()
qc = []
for i in range(len(distributed_points) // 2):
    qc.append(hqga_utils.stato_su_sfera_di_bloch(distributed_points[i * 2], distributed_points[i * 2 + 1]))

base = []
for i in range(len(distributed_points) // 2):
    base.append(DensityMatrix(qc[i]))






# Returns counts

proj=[[state_fidelity(base[j], state_result[i].value)for j in range(params.base)]for i in range(num_var)]
projn=[[proj[i][j]/sum(proj[i]) for j in range(params.base)]for i in range(num_var)]
label=[str(i) for i in range(params.base)]
lista_di_dizionari=[dict(zip(label,projn[i]))for i in range(num_var)]
# Supponiamo che 'dizionario' sia il tuo dizionario di stringhe e probabilità
stringa_selezionata=''
for i in range(num_var):

    # Estrai le chiavi (stringhe) e i valori (probabilità) dal dizionario
    chiavi = list(lista_di_dizionari[i].keys())
    probabilita = list(lista_di_dizionari[i].values())

    # Usa random.choices() per selezionare una chiave basata sulle probabilità
    mis = random.choices(chiavi, weights=probabilita, k=1)[0]
    stringa_selezionata+=str(mis)
print(stringa_selezionata)








'''
# Returns counts

pr = []
for i in range(num_var):
    for j in range(params.base):
        pr.append(state_fidelity(base[j], state_result[i].value))

matrice = [pr[i:i + params.base] for i in range(0, len(pr), params.base)]
label_matrice = [[f"{i}, {j}" for j in range(params.base)] for i in range(num_var)]  # stringa singola
# label_matrice = [[f"{i}, {j:02}" for j in range(p)] for i in range(num_var)] #stringa doppia
num_elementi_da_mantenere = int(params.base * 0.75)  # 75% degli elementi di ogni riga

# Lista per tenere traccia dei valori e dei label mantenuti
valori_mantenuti = []
label_mantenuti = []

for i in range(num_var):
    # Unire valori e label in una lista di tuple
    riga_con_label = list(zip(matrice[i], label_matrice[i]))

    # Ordinare la riga per i valori (dal più alto al più basso)
    riga_ordinata = sorted(riga_con_label, key=lambda x: x[0], reverse=True)

    # Prendere i primi 'num_elementi_da_mantenere' elementi
    riga_top = riga_ordinata[:num_elementi_da_mantenere]

    # Separare i valori e i label
    valori, label = zip(*riga_top)

    # Aggiungere alla lista finale
    valori_mantenuti.append(valori)
    label_mantenuti.append(label)

label_matrice_modificata = [[label.split(", ")[1] for label in riga] for riga in label_mantenuti]
matrice = valori_mantenuti

# Calcolare tutte le combinazioni possibili dei valori dei qubit
combinazioni = list(itertools.product(*matrice))

# Calcolare il prodotto per ciascuna combinazione
prodotti = [hqga_utils.prod(comb) for comb in combinazioni]

# Stampa dei risultati

combinazioni_label = list(itertools.product(*label_matrice_modificata))

lista_di_stringhe = [''.join(tupla) for tupla in combinazioni_label]
# dizionario = dict(zip(lista_di_stringhe, prodotti))

# normalizzazione
prob = prodotti
somma = sum(prob)
prob_n = [elemento / somma for elemento in prob]
#print(prob_n)
#print(sum(prob_n))
dizionario = dict(zip(lista_di_stringhe, prob_n))



# Estrai le chiavi (stringhe) e i valori (probabilità) dal dizionario
chiavi = list(dizionario.keys())
probabilita = list(dizionario.values())
# Usa random.choices() per selezionare una chiave basata sulle probabilità
stringa_selezionata = random.choices(chiavi, weights=probabilita, k=1)[0]
'''
print(type(stringa_selezionata))

# Inserimento di uno spazio ogni due cifre
hk=problem.dim * problem.num_bit_code
stringa_selezionata = ' '.join([stringa_selezionata[i:i+hk] for i in range(0, len(stringa_selezionata), hk)])
print(stringa_selezionata)
counts={}
counts[stringa_selezionata]=1
print("\nCounts:",counts)
print("len counts ", len(counts))

#compute fitness evaluation
classical_chromosomes= hqga_utils.fromQtoC(hqga_utils.getMaxProbKey(counts))
if params.verbose:
    print("\nChromosomes", classical_chromosomes)

l_sup=[]
for c in classical_chromosomes:
    l_sup.append(problem.convert(c))
chromosome_evolution.append(l_sup)
if params.verbose:
    print("Phenotypes:", l_sup)

fitnesses= hqga_utils.computeFitnesses(classical_chromosomes, problem.evaluate)
if params.verbose:
    print("Fitness values:", fitnesses)

best_fitness, index_best= hqga_utils.computeBest(fitnesses, problem.isMaxProblem())
if params.verbose:
    print("Best fitness", best_fitness, "; index best ", index_best)
if gen == 0:
    gBest.chr = classical_chromosomes[index_best]
    gBest.phenotype = problem.convert(gBest.chr)
    gBest.fitness = best_fitness
    gBest.gen = params.pop_size

bests.append([problem.convert(gBest.chr), gBest.fitness, gBest.chr])


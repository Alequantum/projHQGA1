import problems as p, hqga_utils, utils, hqga_algorithm
from utils import computeHammingDistance

from qiskit import Aer
import math

import pandas as pd
from openpyxl import load_workbook
from pathlib import Path





simulator = Aer.get_backend('qasm_simulator')
device_features= hqga_utils.device(simulator, False)

params= hqga_utils.Parameters(base=10,pop_size=3, max_gen=5, prob_mut=0.3,elitism="Quantistic_elitism")
params.draw_circuit=True

problem = p.SphereProblem(num_bit_code=2,base=params.base)

circuit = hqga_utils.setupCircuit(params.pop_size, problem.dim * problem.num_bit_code)




gBest, chromosome_evolution,bests = hqga_algorithm.runQGA(device_features, circuit, params, problem)

#dist=computeHammingDistance(gBest.chr, problem)
#print("The Hamming distance to the optimum value is: ", dist)
#utils.writeBestsXls("Bests.xlsx", bests)
#utils.writeChromosomeEvolutionXls("ChromosomeEvolution.xlsx", chromosome_evolution)

# Il tuo valore di output
output_value = gBest.fitness  # Esempio di valore che vuoi scrivere, cambialo secondo le tue necessità

# Percorso del file Excel (assicurati che il percorso sia corretto)
file_path = 'C:\\Users\\aless\\OneDrive\\Documenti\\outputHQGA_nonortogonale_qubitindipendentitaglio0.70_sfera_base=10_nqubit=2_corretto.xlsx'

# Crea un nuovo DataFrame con il valore da aggiungere
new_row = pd.DataFrame([output_value], columns=['Valore'])

# Controlla se il file esiste e non è corrotto
try:
    # Tenta di caricare il workbook esistente
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}. Un nuovo file verrà creato.")
    df = pd.DataFrame()  # Crea un DataFrame vuoto se il file non esiste o non può essere letto

# Aggiungi il nuovo valore al DataFrame esistente
df = pd.concat([df, new_row], ignore_index=True)

# Salva il DataFrame nel file Excel
try:
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
except Exception as e:
    print(f"Si è verificato un errore durante il salvataggio del file: {e}")




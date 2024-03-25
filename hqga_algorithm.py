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
def runQGA(device_features,circuit, params,problem):
    """Function that runs HQGA

    Args:
        devices_features (hqga_utils.device): information about the device used to run HQGA
        circuit (qiskit.QuantumCircuit): circuit to be run during HQGA iterations
        params (hqga_utils.Parameters): information about the hyper-parameters of HQGA
        problem (problems.Problem): information about the problem to be solved

    Returns:
        gBest (hqga_utils.globalBest): object that stores the information about the best solution found during the evolution
        chromosome_evolution (list): object that stores the population generated in each iteration
        bests (list): object that stores the best solutions found during each iteration
    """
    num_var=params.pop_size * problem.dim * problem.num_bit_code #numero di qubit del circuito
    chromosome_evolution=[]
    bests =[]
    #gen=0
    theta = hqga_utils.initializeTheta(circuit)
    gBest = hqga_utils.globalBest()
    list_qubit_gate, list_qubit_entang, list_qubit_mutation, list_qubit_X= hqga_utils.initializeLists(circuit)
    dict_chr=[]
    flag_index_best=False
    #while gen<=max_gen:
    l_gen=range(params.max_gen+1)
    if params.progressBar:
        l_gen= tqdm(range(params.max_gen+1), desc="Generations")


    distributed_points = hqga_utils.distribute_points_on_sphere(params.base).flatten().tolist()
    qc = []
    for i in range(len(distributed_points) // 2):
        qc.append(hqga_utils.stato_su_sfera_di_bloch(distributed_points[i * 2], distributed_points[i * 2 + 1]))

    base = []
    for i in range(len(distributed_points) // 2):
        base.append(DensityMatrix(qc[i]))

    for gen in l_gen:
        #print("\n########## generation #########", gen)
        hqga_utils.applyMultiHadamardOnList(circuit, list_qubit_gate)
        hqga_utils.applyMultiRotationOnList(circuit, theta, list_qubit_gate)
        hqga_utils.applyXOnList(circuit, list_qubit_X, dict_chr,qc)
        if gen!=0:
            circuit.barrier()
            hqga_utils.applyEntanglementOnList(circuit, index_best, list_qubit_entang, theta)
            circuit.barrier()
            hqga_utils.applyMutationOnListWithinRange(circuit, params.prob_mut, list_qubit_mutation, theta)
        circuit.barrier()

        #hqga_utils.applyMeasureOperator(circuit)
        # Draw the circuit
        if params.draw_circuit:
            print(circuit.draw(output="text", fold=300))
            print("Circuit depth is ",circuit.depth())

        # Execute the circuit on the qasm simulator
        while True:
            circuit.name =str(params.qobj_id)+str(gen)
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







        # Returns counts

        proj = [[state_fidelity(base[j], state_result[i].value) for j in range(params.base)] for i in range(num_var)]
        #projn = [[proj[i][j] / sum(proj[i]) for j in range(params.base)] for i in range(num_var)]
        label = [str(i) for i in range(params.base)]
        #lista_di_dizionari = [dict(zip(label, projn[i])) for i in range(num_var)]
        lista_di_dizionari = [dict(zip(label, proj[i])) for i in range(num_var)]


        #Creazione dizionario ridotto: per ogni qubit considero solo i valori piu probabili
        lista_dizionari_ridotto = []
        for i in range(num_var):
            dizionario = lista_di_dizionari[i]
            # Ordiniamo il dizionario in base ai valori, dal più alto al più basso
            dizionario_ordinato = dict(sorted(dizionario.items(), key=lambda item: item[1], reverse=True))
            # Calcoliamo il 75% degli elementi da conservare
            elementi_da_conservare = int(len(dizionario_ordinato) * 0.70)
            # Conserviamo solo il 75% degli elementi
            dizionario_ridotto = dict(list(dizionario_ordinato.items())[:elementi_da_conservare])
            lista_dizionari_ridotto.append(dizionario_ridotto)

        #normalizziamo le probabilita su ogni qubit
        # Calcoliamo la somma dei valori nel dizionario ridotto
        dizionario_ridotto_normalizzato = []
        for i in range(num_var):
            somma_valori = sum(lista_dizionari_ridotto[i].values())
            # Normalizziamo i valori affinché sommino a 1
            dizionario_normalizzato = {chiave: valore / somma_valori for chiave, valore in
                                       lista_dizionari_ridotto[i].items()}
            dizionario_ridotto_normalizzato.append(dizionario_normalizzato)
        lista_di_dizionari = dizionario_ridotto_normalizzato


        # Supponiamo che 'dizionario' sia il tuo dizionario di stringhe e probabilità
        stringa_selezionata = ''
        for i in range(num_var):
            # Estrai le chiavi (stringhe) e i valori (probabilità) dal dizionario
            chiavi = list(lista_di_dizionari[i].keys())
            probabilita = list(lista_di_dizionari[i].values())
            # Usa random.choices() per selezionare una chiave basata sulle probabilità
            mis = random.choices(chiavi, weights=probabilita, k=1)[0]
            stringa_selezionata += str(mis)










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




        '''
        s = hqga_utils.generate_input_strings(num_var, params.base)
        # print(s)  # Stampa le prime 10 stringhe di input
        pr = []
        for i in range(params.base ** num_var):
            for j in range(num_var):
                # pr.append(state_fidelity(base[int(s[i][2*j:2*j+2])], state_result[j].value))#stringa doppia
                pr.append(state_fidelity(base[int(s[i][j])], state_result[j].value))  # stringa singola
        # Creare una nuova lista per i prodotti
        prob = []
        # Iterare attraverso la lista 'p' a gruppi di 4 elementi
        # Iterare attraverso la lista 'pr' a gruppi di 'num_var' elementi
        for i in range(0, len(pr), num_var):
            prodotto = 1
            for j in range(num_var):
                prodotto *= pr[i + j]
            prob.append(prodotto)
        # Ora 'prob' contiene il prodotto di ogni gruppo di 4 elementi in 'p'
        somma = sum(prob)
        prob_n = [elemento / somma for elemento in prob]
        dizionario = dict(zip(s, prob_n))
        
        # Estrai le chiavi (stringhe) e i valori (probabilità) dal dizionario
        chiavi = list(dizionario.keys())
        probabilita = list(dizionario.values())
        # Usa random.choices() per selezionare una chiave basata sulle probabilità
        stringa_selezionata = random.choices(chiavi, weights=probabilita, k=1)[0]
        '''




        # Inserimento di uno spazio ogni due cifre
        hk = problem.dim * problem.num_bit_code
        stringa_selezionata = ' '.join([stringa_selezionata[i:i + hk] for i in range(0, len(stringa_selezionata), hk)])
        counts={}
        counts[stringa_selezionata]=1
        #print("\nCounts:",counts)
        #print("len counts ", len(counts))

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
        if gen!=0:
            previous_best=index_best
        best_fitness, index_best= hqga_utils.computeBest(fitnesses, problem.isMaxProblem())
        if params.verbose:
            print("Best fitness", best_fitness, "; index best ", index_best)
        if gen == 0:
            gBest.chr = classical_chromosomes[index_best]
            gBest.phenotype = problem.convert(gBest.chr)
            gBest.fitness = best_fitness
            gBest.gen = params.pop_size
        else:
            flag_index_best = previous_best != index_best
            if hqga_utils.isBetter(best_fitness, gBest.fitness, problem.isMaxProblem()):
                gBest.chr = classical_chromosomes[index_best]
                gBest.phenotype = problem.convert(gBest.chr)
                gBest.fitness = best_fitness
                gBest.gen = (gen+1)*params.pop_size
        bests.append([problem.convert(gBest.chr), gBest.fitness, gBest.chr])


        list_qubit_gate, list_qubit_entang,list_qubit_mutation,list_qubit_X= hqga_utils.computeLists(circuit, index_best, params.pop_size, problem.dim * problem.num_bit_code)

        if params.elitism is not hqga_utils.ELITISM_Q:
            #update
            dict_chr = hqga_utils.create_dict_chr(circuit, classical_chromosomes)

            if params.elitism is hqga_utils.ELITISM_R:
                if flag_index_best:
                    hqga_utils.resetThetaReinforcement(dict_chr, theta, old_theta, previous_best, problem.dim*problem.num_bit_code)
                old_theta=copy.deepcopy(theta)
                hqga_utils.updateThetaReinforcementWithinRange(dict_chr, theta, params.epsilon, index_best, problem.dim*problem.num_bit_code)
            elif params.elitism is hqga_utils.ELITISM_D:
                hqga_utils.updateListXElitismD(circuit, index_best, list_qubit_gate, list_qubit_X)
            else:
                raise Exception("Value for elitism is not valid.")

        hqga_utils.resetCircuit(circuit)
        #gen+=1

    gBest.display()
    print("The number of fitness evaluations is: ", params.pop_size*(params.max_gen+1))
    return gBest, chromosome_evolution,bests

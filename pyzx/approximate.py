import sys, os; sys.path.insert(0, '..')
import random, logging, math
import pyzx as zx
from fractions import Fraction
from pyzx.gadget_extract import *
import matplotlib.pyplot as plt
import numpy as np
zx.settings.drawing_backend = 'matplotlib'

def graph_rounder(g:zx.graph.graph_s.GraphS, vertex_flip, vertex_list, rounded_phase_dict, phase_dict=[], already_rounded: bool = False) -> zx.graph.graph_s.GraphS:
    '''
    Takes in a graph, its vertices and the ones to flip.
    If already rounded is false then the graph is assumed to be in its completely unrounded state and so does not need the original phase_dict.
    If already rounded is true then the graph may need to unround some vertices so will do this. 
    '''
    for v in vertex_list:
        if vertex_flip[v]:
            g.set_phase(v, rounded_phase_dict[v])
        elif already_rounded:
            g.set_phase(v,phase_dict[v])
    return g

def accept_prob(new_count,count,count_init,temp):
    '''
    Generate an acceptance probability.
    '''
    delta = (new_count-count)/count_init
    prob = math.exp(-delta/temp)
    return prob

def approximation_dicts(g):
    '''
    Return the set of dictionaries used in phase squashing.
    '''
    vertex_list = [v for v in list(g.vertices()) if (g.phase(v).denominator > 2)] # Get all non-Clifford vertices
    phase_dict = {v:(g.phase(v)) for v in vertex_list} # The original phases of all non-Clifford gates
    rounded_phase_dict = {v:(Fraction(round(2*g.phase(v)), 2)) for v in vertex_list} # The rounded version of all non-Clifford gates
    round_error_dict = {v:abs(float(phase_dict[v])- rounded_phase_dict[v]) for v in vertex_list} # List of errors between a gate and its nearest pi/2
    return vertex_list, phase_dict, rounded_phase_dict, round_error_dict

def approximate_anneal(g: zx.graph.graph_s.GraphS, err_budget: float, quiet:bool =True, metric: str = 'gates', cold=False, cooling_rate:float = 0.6) -> float:
    # General parameters
    total_err = 0
    max_iter = 1000

    # Temperature parameters
    temp = 1
    temp_array = [temp]
    accetpance_prob_array = [1]

    # Initialise counters
    change_counter = 0
    no_change_counter = 0
    iter = 0
    freeze_out_counter = 0

    # Initial count
    c = zx.extract_circuit(g.clone())
    count = c.stats_dict()[metric]
    count_init = count
    count_list = [count]
    err_array = [0]

    vertex_list, phase_dict, rounded_phase_dict, round_error_dict = approximation_dicts(g)
    vertex_flip = {v:False for v in vertex_list}

    # Remove vertices that would break the error budget if chosen
    vertices_too_big = [i for i in vertex_list if (total_err + 2*abs(math.sin(np.pi * round_error_dict[i]/2))) > err_budget]
    vertex_list = [i for i in vertex_list if i not in vertices_too_big]
    if not vertex_list: # If all verticies would break the error budget
        if not quiet: print("All gates would break error budget, nothing to squash.")
        return 0
    
    # Create probability list
    inv_error = [round_error_dict[v]**(-1) for v in vertex_list]
    prob_to_be_picked = [i/sum(inv_error) for i in inv_error]
    freeze_out_max = len(vertex_list) * 2
    epoch = len(vertex_list)
    new_best_reached = False

    # Greedy algorithm to set up initial state
    sorted_vertex_list = sorted(rounded_phase_dict, key=lambda x: round_error_dict[x])
    for v in sorted_vertex_list:
        g_clone = g.clone()
        vertex_flip[v] = True
        graph_rounder(g_clone, vertex_flip,vertex_list,rounded_phase_dict)
        zx.full_reduce(g_clone)
        c_test = zx.extract_circuit(g_clone)
        new_count = c_test.stats_dict()[metric]
        if new_count < count:
            norm_err = 2*abs(math.sin(np.pi * round_error_dict[v]/2))
            if total_err + norm_err <= err_budget:
                total_err += norm_err
                count = new_count
            else:
                vertex_flip[v] = False
        else:
            vertex_flip[v] = False

    vertex_flip_best = vertex_flip.copy()
    count_best = count

    while iter<=max_iter and freeze_out_counter <= freeze_out_max and count !=0: # while error is within budget, max-iterations is not surpassed and we have not frozen
        iter_at_temp = 1
        while True:
            # CHOOSE RANDOM VERTEX
            v = np.random.choice(vertex_list,p=prob_to_be_picked)

            # CHECK THE VERTEX WILL NOT BREACH ERROR BUDGET
            # TODO perhaps allow breakages during the annealing process as long as eventually comes back under budget
            norm_err = 2*abs(math.sin(np.pi * round_error_dict[v]/2))
            if (not vertex_flip[v]) and (total_err + norm_err) > err_budget:
                freeze_out_counter += 1
            
            else:
                # TEST GRAPH
                vertex_flip[v] = not vertex_flip[v]
                g_test = g.clone()
                graph_rounder(g_test, vertex_flip, vertex_list, rounded_phase_dict, phase_dict=phase_dict, already_rounded=True)
                zx.full_reduce(g_test)
                c_test = zx.extract_circuit(g_test)
                count_new = c_test.stats_dict()[metric]

                if count_new > count:
                    accetpance_prob_array.append(accept_prob(count_new, count, count_init, temp))
                else:
                    accetpance_prob_array.append(accetpance_prob_array[-1])

                # BETTER RESULT ACCEPTANCE
                if count_new < count: # If this gives a better solution then we automatically accept it
                    change_counter += 1
                    count = count_new
                    count_list.append(count)

                    if count < count_best:
                        vertex_flip_best = vertex_flip.copy()
                        count_best = count
                        new_best_reached = True

                    # Change the total error bound
                    if not vertex_flip[v]: # If we are unrounding (hence we are undoing the approximation) then reduce error
                        total_err -= norm_err
                        err_array.append(total_err)
                        freeze_out_counter = 0

                    else: # And if we are rounding, add the error
                        total_err += norm_err
                        err_array.append(total_err)
                        freeze_out_counter = 0

                    # If we have reached zero two qubit gates, quit
                    if count == 0:
                        break                

                # SAME GATE COUNT ACCEPTANCE
                elif count_new==count and np.random.random() < math.exp(-1/(count_init*temp)) and not cold: # If a worse solution but not TOO much worse, then maybe accept
                    change_counter += 1
                    count_list.append(count)

                    if not vertex_flip[v]: # If we are unrounding (hence we are undoing the approximation) then reduce error
                        total_err -= norm_err
                        err_array.append(total_err)
                        freeze_out_counter = 0

                    else: # And if we are rounding, add the error
                        total_err += norm_err
                        err_array.append(total_err)
                        freeze_out_counter = 0
                
                # POTENTIAL NON-OPTIMAL ACCEPTANCE
                elif count_new>count and np.random.random() < accept_prob(count_new, count,count_init, temp) and not cold: # If a worse solution but not TOO much worse, then maybe accept
                    change_counter += 1
                    count = count_new
                    count_list.append(count)

                    if not vertex_flip[v]: # If we are unrounding (hence we are undoing the approximation) then reduce error
                        total_err -= norm_err
                        err_array.append(total_err)
                        freeze_out_counter = 0

                    else: # And if we are rounding, add the error
                        total_err += norm_err
                        err_array.append(total_err)
                        freeze_out_counter = 0

                # NO CHANGE ACCEPTED
                else:
                    vertex_flip[v] = not vertex_flip[v]
                    no_change_counter +=1
                    freeze_out_counter += 1
                    count_list.append(count_list[-1])
                    err_array.append(err_array[-1])

            iter += 1
            iter_at_temp += 1
            temp_array.append(temp)
            if iter>max_iter:
                break

            # CHECK EQUILIBRIUM
            if iter_at_temp%epoch == 0: # Every epoch
                if new_best_reached: # If we found a new minima
                    new_best_reached = False # Maintain temperature for next epoch
                else:
                    break # Quit current epoch

        temp *= cooling_rate # Cool the system

    # OPTIMIZING FINAL ARRANGMENT
    graph_rounder(g, vertex_flip_best, vertex_list, rounded_phase_dict)
    changed_check , vertex_flip_deapproximated, err_change = deapprox_redundancies(g, vertex_flip_best, phase_dict, rounded_phase_dict, round_error_dict, vertex_flip)
    if changed_check:
        total_err += err_change
        graph_rounder(g, vertex_flip_deapproximated, vertex_list,rounded_phase_dict,phase_dict=phase_dict, already_rounded=True)
        zx.full_reduce(g)    
        approximate(g,err_budget=err_budget,current_err=total_err, metric=metric)

    # DIAGNOSTICS
    if not quiet:
        if total_err > err_budget:
            print("Error Exceeded")
        if iter>max_iter:
            print("Max Iterations Exceeded")
        if freeze_out_counter > freeze_out_max:
            print("System Froze")
        if count==0:
            print('Two-qubit Count Reached 0')

        print(f'Total error: {total_err}')
        print(f"{sum(vertex_flip.values())} / {len(vertex_list)} gates squashed")
        print(f'Num changes: {change_counter}/{iter}, Num no changes: {no_change_counter}/{iter}')
        plt.style.use('bmh')
        fig, ax = plt.subplots(4)
        ax[0].set(ylabel='Temperature')
        ax[0].plot(temp_array)
        #ax[0].set_xscale('log')
        ax[1].set(ylabel='Metric Count')
        #ax[1].set_xscale('log')
        ax[1].plot(count_list)
        ax[2].plot(accetpance_prob_array)
        ax[2].set(ylabel='Acceptance Probability')
        ax[3].plot(err_array)
        ax[3].set(ylabel='Error')
        fig.set_figheight(10)
        fig.set_figwidth(10)

    zx.full_reduce(g,quiet=True)
    return total_err

def approximate(g: zx.graph.graph_s.GraphS, err_budget:float, current_err:float = 0, quiet:bool = True, metric: str = 'gates')->float:
    vertex_list, phase_dict, rounded_phase_dict, round_error_dict = approximation_dicts(g)
    total_err = current_err
    count = zx.extract_circuit(g.clone()).stats_dict()[metric]
    calls_to_round = 0
    count_list = [count]
    sorted_vertex_list = sorted(rounded_phase_dict, key=lambda x: round_error_dict[x])
    for v in sorted_vertex_list:
        g_clone = g.clone()
        g_clone.set_phase(v,rounded_phase_dict[v])
        zx.full_reduce(g_clone)
        c_test = zx.extract_circuit(g_clone)
        new_count = c_test.stats_dict()[metric]
        calls_to_round += 1
        if new_count < count:
            norm_err = 2*abs(math.sin(np.pi * round_error_dict[v]/2))
            if total_err + norm_err <= err_budget:
                total_err += norm_err
                count = new_count
                count_list.append(count)
                g.set_phase(v,rounded_phase_dict[v])
        else:
            count_list.append(count_list[-1])
    if not quiet:
        plt.style.use("bmh")
        plt.plot(count_list)
    zx.full_reduce(g)
    return total_err

def approximate_basic(g: zx.graph.graph_s.GraphS, err_budget: float, current_err:float=0, quiet: bool =True, metric: str = 'gates')->float:
    vertex_list, phase_dict, rounded_phase_dict, round_error_dict = approximation_dicts(g)
    total_err = current_err
    count = zx.extract_circuit(g.clone()).stats_dict()[metric]
    calls_to_round = 0
    count_list = [count]
    for v in vertex_list:
        g_clone = g.clone()
        g_clone.set_phase(v,rounded_phase_dict[v])
        basic_simp_fast(g_clone)
        c_test = zx.extract_circuit(g_clone)
        new_count = c_test.stats_dict()[metric]
        calls_to_round += 1
        if new_count - count <0:
            norm_err = 2*abs(math.sin(np.pi * round_error_dict[v]/2))
            if total_err + norm_err <= err_budget:
                total_err += norm_err
                count = new_count
                count_list.append(count)
                g.set_phase(v,rounded_phase_dict[v])
        else:
            count_list.append(count_list[-1])
    if not quiet:
        plt.style.use("bmh")
        plt.plot(count_list)
    basic_simp_fast(g)
    return total_err

def deapprox_redundancies(g: zx.graph.graph_s.GraphS, vertex_list, phase_dict, rounded_phase_dict, round_error_dict, vertex_flip, metric = 'gates'):
    '''
    This runs through each of the rounded vertices and unrounds them, if the resulting circuit has the same number of
    vertices then it undos the rounding to minimise the error.
    '''
    g_init = g.clone()
    zx.full_reduce(g_init)
    c = zx.extract_circuit(g_init)
    count = c.stats_dict()[metric]
    vertex_list_rounded = [v for v in vertex_list if vertex_flip[v]]
    changed_check = False
    vertex_flip_deapproximated = vertex_flip.copy()
    err_change = 0
    for v in vertex_list_rounded:
        vertex_flip_test = vertex_flip.copy()
        vertex_flip_test[v] = not vertex_flip_test[v]
        g_test = g.clone()
        graph_rounder(g_test, vertex_flip_test, vertex_list, rounded_phase_dict, phase_dict, already_rounded=True)
        zx.full_reduce(g_test)
        c = zx.extract_circuit(g_test)
        new_count = c.stats_dict()[metric]
        if new_count <= count:
            changed_check = True
            vertex_flip_deapproximated[v] = not vertex_flip_deapproximated[v]
            err_change -= 2*abs(math.sin(np.pi * round_error_dict[v]/2))
    return changed_check, vertex_flip_deapproximated, err_change

def basic_simp_fast(g) -> int:
    """Keeps doing the simplifications ``id_simp`` and ``spider_simp`` until none of them can be applied anymore. If
    starting from a circuit, the result should still have causal flow."""
    zx.simplify.to_gh(g, quiet=True)
    i = 0
    while True:
        i1 = zx.simplify.id_simp(g, quiet=True)
        i2 = zx.simplify.spider_simp(g, quiet=True)
        if i1+i2==0: break
        i += 1

def two_qubit_gate_count(g: zx.graph.graph_s.GraphS) -> int:
    num_v = g.num_vertices()
    num_e = g.num_edges()
    num_qubits = g.num_inputs()
    two_qubit_gates = num_e - num_v + num_qubits
    return(two_qubit_gates)

def two_qubit_basic_anneal(g: zx.graph.graph_s.GraphS, err_budget: float, max_iter, cooling_rate=0.5, quiet:bool =True):
    '''
    Adaptive temperature schedule:
    We set an epoch, eg. number of vertices.
    after each epoch we check to see if a new minima was found, if not then we cool the system.
    '''
    # General parameters
    total_err = 0
    tq = two_qubit_gate_count(g)
    tq_list = [tq]
    tq_init = tq
    accetpance_prob_array = []
    tq_best = float("inf")
    new_best_reached = False

    # Temperature parameters
    cooling_rate = cooling_rate
    temp = 1
    temp_array = [temp]
    accetpance_prob_array = [1]

    # Initialise counters
    change_counter = 0
    no_change_counter = 0
    iter = 0
    freeze_out_counter = 0

    # Create all of the graphs we could need
    # TODO is this really the most efficient way? Many redudancies in the case of a freeze-out.
    g_dict = {}
    for i in range(max_iter+1):
        g_dict[f'g_{i}'] = g.clone()

    # Initialise the graphs non-Clifford vertices
    vertex_list = [v for v in list(g.vertices()) if (g.phase(v).denominator > 2)] # Get all non-Clifford vertices
    vertex_flip = {v:False for v in vertex_list} # Checks if vertex has been rounded, set all initially to False
    phase_dict = {v:(g.phase(v)) for v in vertex_list} # The original phases of all non-Clifford gates
    rounded_phase_dict = {v:(Fraction(round(2*g.phase(v)), 2)) for v in vertex_list} # The rounded version of all non-Clifford gates
    round_error_dict = {v:abs(float(phase_dict[v])- rounded_phase_dict[v]) for v in vertex_list} # List of errors between a gate and its nearest pi/2
    best_vertex_flip = vertex_flip.copy()

    # Remove vertices that would break the error budget if chosen
    vertices_too_big = [i for i in vertex_list if (total_err + 2*abs(math.sin(np.pi * round_error_dict[i]/2))) > err_budget]
    vertex_list = [i for i in vertex_list if i not in vertices_too_big]
    if not vertex_list: # If all verticies would break the error budget
        print("All gates would break error budget, nothing to squash.")
        return
    
    # Create probability list
    inv_error = [round_error_dict[v]**(-1) for v in vertex_list] # Invert error, used to calculate probability of being picked
    prob_to_be_picked = [i/sum(inv_error) for i in inv_error] # Probability of vertex to be chosen, smaller errors are more likely

    # Set the number of iterations we can have without a change to be equal to the number of changable vertices
    freeze_out_max = len(vertex_list) * 5
    epoch = len(vertex_list) 
    
    while (total_err <= err_budget) and (iter<=max_iter) and freeze_out_counter <= freeze_out_max and tq !=0: # while error is within budget, max-iterations is not surpassed and we have not frozen
        iter_at_temp = 1
        while True:
            # CHOOSE RANDOM VERTEX
            v = np.random.choice(vertex_list, p=prob_to_be_picked)

            # CHECK THE VERTEX WILL NOT BREACH ERROR BUDGET
            # TODO perhaps allow breakages during the annealing process as long as eventually comes back under budget
            norm_err = 2*abs(math.sin(np.pi * round_error_dict[v]/2))
            if (not vertex_flip[v]) and (total_err + norm_err) > err_budget:
                freeze_out_counter += 1
                continue

            # TEST GRAPH
            vertex_flip[v] = not vertex_flip[v]
            graph_rounder(g_dict[f'g_{iter}'], vertex_flip, vertex_list, rounded_phase_dict)
            basic_simp_fast(g_dict[f'g_{iter}']),
            tq_new = two_qubit_gate_count(g_dict[f'g_{iter}'])

            # BETTER RESULT ACCEPTANCE
            if tq_new < tq: # If this gives a better solution the we automatically accept it
                accetpance_prob_array.append(accetpance_prob_array[-1])
                change_counter += 1

                # Change the total error bound
                if not vertex_flip[v]: # If we are unrounding (hence we are undoing the approximation) then reduce error
                    total_err -= norm_err
                    freeze_out_counter = 0

                else: # And if we are rounding, add the error
                    total_err += norm_err
                    freeze_out_counter = 0

                # CHECK FOR NEW MINIMA
                if tq_new < tq_best:
                    # TODO for some reason we have to set the best graph here, we can not wait till the end and use a stored best_vertex_flip dictionary.
                    # this may give performance issues since the graph is being rewritten more than it should but it seems minimal for now. 
                    best_vertex_flip = {v:vertex_flip[v] for v in vertex_list}
                    graph_rounder(g,best_vertex_flip, vertex_list, rounded_phase_dict,phase_dict, already_rounded=True)
                    tq_best = tq
                    new_best_reached = True
                
                tq = tq_new
                tq_list.append(tq_new)

                # If we have reached zero two qubit gates, quit
                if tq == 0:
                    break
            
            # POTENTIAL NON-OPTIMAL ACCEPTANCE
            elif tq_new>tq and (np.random.random() < accept_prob(tq_new, tq, tq_init, temp)): # If a worse solution but not TOO much worse, then maybe accept
                accetpance_prob_array.append(accept_prob(tq_new, tq, tq_init, temp))
                tq_list.append(tq_new)
                change_counter += 1
                tq = tq_new

                if not vertex_flip[v]: # If we are unrounding (hence we are undoing the approximation) then reduce error
                    total_err -= norm_err
                    freeze_out_counter = 0

                else: # And if we are rounding, add the error
                    total_err += norm_err
                    freeze_out_counter = 0

            # NO CHANGE ACCEPTED
            else:
                accetpance_prob_array.append(accetpance_prob_array[-1])
                tq_list.append(tq_list[-1])
                vertex_flip[v] = not vertex_flip[v]
                no_change_counter +=1
                freeze_out_counter += 1

            iter += 1
            iter_at_temp += 1
            temp_array.append(temp)
            if iter>max_iter:
                break

            # CHECK EQUILIBRIUM
            if iter_at_temp%epoch == 0: # Every epoch
                if new_best_reached: # If we found a new minima
                    new_best_reached = False # Maintain temperature for next epoch
                else:
                    break # Quit current epoch

        temp *= cooling_rate # Cool the system

    # OPTIMIZING FINAL ARRANGMENT
    graph_rounder(g, best_vertex_flip, vertex_list,rounded_phase_dict)
    changed_check , vertex_flip_deapproximated, err_change = deapprox_redundancies(g, best_vertex_flip, phase_dict, rounded_phase_dict, round_error_dict, vertex_flip)
    if changed_check:
        total_err += err_change

    # DIAGNOSTICS
    if not quiet:
        if total_err > err_budget:
            print("Error Exceeded")
        if iter>max_iter:
            print("Max Iterations Exceeded")
        if freeze_out_counter > freeze_out_max:
            print("System Froze")
        if tq==0:
            print('Two-qubit Count Reached 0')

        print(f'Total error: {total_err}')
        print(f"{sum(vertex_flip.values())} / {len(vertex_list)} gates squashed")
        print(f'Num changes: {change_counter}/{iter}, Num no changes: {no_change_counter}/{iter}')
        plt.style.use('ggplot')
        fig, ax = plt.subplots(3)
        ax[0].set(ylabel='Temperature')
        ax[0].plot(temp_array)
        #ax[0].set_xscale('log')
        ax[1].set(ylabel='tq gates')
        #ax[1].set_xscale('log')
        ax[1].plot(tq_list)
        ax[2].plot(accetpance_prob_array)
        ax[2].set(ylabel='Acceptance Probability')
        fig.set_figheight(20)
        fig.set_figwidth(10)
    
    zx.simplify.basic_simp(g,quiet=True)
    return total_err

def greedy_basic_reduce(g,err_budget):
    total_err = 0
    basic_simp_fast(g)
    vertex_list, phase_dict, rounded_phase_dict, round_error_dict = approximation_dicts(g)
    sorted_vertex_list = sorted(rounded_phase_dict, key=lambda x: round_error_dict[x])
    count = two_qubit_gate_count(g)
    for v in sorted_vertex_list:
        norm_err = 2*abs(math.sin(np.pi * round_error_dict[v]/2))
        if total_err + norm_err < err_budget:
            g_test = g.clone()
            g_test.set_phase(v,rounded_phase_dict[v])
            basic_simp_fast(g_test)
            new_count = two_qubit_gate_count(g_test)
            if new_count < count:
                total_err += norm_err
                g.set_phase(v,rounded_phase_dict[v])
        else:
            break
    basic_simp_fast(g)
    return total_err
    
def greedy_reduce(g: zx.graph.graph_s.GraphS, err_budget, quiet=True) -> float:
    total_err = 0
    # Start with the closest approximations, then work our way down to
    # progressively less accurate approximations until either the error
    # budget is exceeded or all the non-Clifford phases have been squashed.
    for i in range(50, 1, -1):
        zx.full_reduce(g)
        #th = Fraction(1, 2**i) # Threshold goes from pi/2^50 to pi/4 in exponentially larger jumps
        #if not quiet: print(f'threshold Δθ <= 2^-{i} π')
        th = Fraction(1,4) - Fraction(1,48*4) * (i-2) # Threshold goes from 0 to pi/4 in even steps (note, this is a really ugly way to write this but it allows it be under the same loop as the other th)
        if not quiet: print(f'threshold Δθ <= {th} π')
        for v in list(g.vertices()):
            p = g.phase(v)
            if p.denominator > 2:  # If the phase is non-Clifford
                new_p = Fraction(round(2*p), 2)  # Squash the phase by rounding to the nearest multiple of 1/2
                err = abs(float(p)-(new_p)) # TODO: Ask why the new_p is rounded (I have since removed this)
                if err <= th:  # If the squashed phase is close enough to the original (as determined by threshold)
                    if total_err + err > err_budget:
                        zx.full_reduce(g)
                        if not quiet: print('exceeded error budget, done.')
                        return
                    else:
                        norm_err = 2*abs(math.sin(np.pi * err/2)) # This equation comes from Eq.3.9 on pg.36 this thesis: https://www.cs.ox.ac.uk/people/aleks.kissinger/theses/agnel-thesis.pdf 
                        if not quiet: print(f' * {p} ~ {new_p} (ε = {norm_err})')
                        g.set_phase(v, new_p)  # Replace the phase with the squashed phase
                        total_err += norm_err  # Keep track of total error
    zx.full_reduce(g)
    return total_err

def qft_two_qubit_approximation(g,err_budget):
    squashable_vertices = c_phase_squashable_gates(g)
    count = two_qubit_gate_count(g)
    vertex_list, phase_dict, rounded_phase_dict, round_error_dict = approximation_dicts(g)
    sorted_vertex_list = sorted(squashable_vertices, key=lambda x: round_error_dict[x])
    total_err = 0
    for v in sorted_vertex_list:
        g_test = g.clone()
        norm_err = 2*abs(math.sin(np.pi * round_error_dict[v]/2))
        if norm_err + total_err < err_budget:
            g_test.set_phase(v,rounded_phase_dict[v])
            basic_simp_fast(g_test)
            new_count = two_qubit_gate_count(g_test)
            if new_count < count:
                count = new_count
                total_err += norm_err
                g.set_phase(v,rounded_phase_dict[v])
    basic_simp_fast(g)
    return total_err

def c_phase_squashable_gates(g):
    zx.to_gh(g)
    basic_simp_fast(g)
    edge_list = [e for e in g.edges()]
    non_zero_vertices = [v for v in g.vertices() if g.phase(v) < 0.5 or g.phase(v) > 1.5]
    squashable_vertices = []
    for v in non_zero_vertices:
        neighbors = list(g.neighbors(v))
        if len(neighbors) == 2:
            next_but_one = set(g.neighbors(neighbors[0]))
            next_but_two = set(g.neighbors(neighbors[1]))
            next_but_one.remove(v)
            next_but_two.remove(v)
            if not next_but_one.isdisjoint(next_but_two):
                squashable_vertices.append(v)
    return squashable_vertices
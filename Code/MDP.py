"""
***********************************************
* LINFO2275 : Data mining and decision making *
* ------------------------------------------- *
*                                             *
*      MDP solving with value iteration       *
*            Project #1, 2022                 *
*                                             *
***********************************************

@author:    Aymeric WARNAUTS 
            Nathan NEPPER
            Tanguy LOSSEAU 

@program:   SINF2M
            DATS2M
            INFO2M

@group:     nº22 

"""

import sys
import random
import csv
import numpy as np


class Action:
    names = ["secu", "normal", "risky"]
    ranges = {"secu": 1, "normal": 2, "risky": 3}
    trigger_probability = {"secu": 0, "normal": 0.5, "risky": 1}


class Consequence:
    def __init__(self, cost, probability, end_state):
        self.cost = cost
        self.probability = probability
        self.end_state = end_state

    def __str__(self):
        return f"(c:{self.cost}, p:{self.probability:.2f}, end:{self.end_state.position})"


class State:
    def __init__(self, position, category):
        self.position = position
        self.category = category

        self.results_of = {}
        self.successors = []
        self.predecessor = None
        self.states = None


def build_states(layout, looping):
    """ 
    Build every States with their various possibilities of Consequence and their associated probability
    -------------------------------------------------------------
    Inputs :
        - layout  :  numpy ndarray. 
                    layout of the traps on the board
        - looping : boolean.
                    if True, you have to land exactly on square 15 to win, 
                    otherwise you restart from the beginning.
    -------------------------------------------------------------
    Outputs :
        - states : List
                   List of every States with their various attribute 
                   based on the layout and the type of board.
    """
    if layout[0] != 0 or layout[-1] != 0:
        print("Invalid layout")
        return

    states = [State(i, layout[i-1]) for i in range(1, 16)]

    # Connect the states
    for i in range(len(states)):

        states[i].predecessor = states[0] if i == 0 else states[i - 1]

        if not looping:
            states[i].successors = [states[14]] if i == 14 else [states[i + 1]]
        else:
            states[i].successors = [states[0]] if i == 14 else [states[i + 1]]

        if i == 2:
            states[i].successors = [states[3], states[10]]
        elif i == 9:
            states[i].successors = [states[14]]
        elif i == 10:
            states[i].predecessor = states[2]

    # Compute for each state, it's actions consequences.
    for state in states:
        state.states = states
        for action in Action.names:

            # Find "connected" nodes
            if state.position == 15:
                state.results_of[action] = [Consequence(0, 1, state)]

            else:
                exploring = state.successors
                found = [state, state] if len(state.successors) > 1 else [state]

                for step in range(Action.ranges[action]):
                    next_expo = []
                    for explored in exploring:
                        found.append(explored)
                        next_expo.append(explored.successors[0])
                    exploring = next_expo

                proba = 1/len(found)
                consequences = [Consequence(1, proba, end_state) for end_state in found]

                #Apply traps.
                trap_results = []
                if Action.trigger_probability[action] > 0:
                    for i in range(len(consequences)):
                        trap_consequence = None

                        if consequences[i].end_state.category == 0:
                            # Basic end_state: nothing to modify.
                            continue
                        elif consequences[i].end_state.category == 1:
                            # Trap: return to position 0
                            trap_consequence = Consequence(1, consequences[i].probability, states[0])
                        elif consequences[i].end_state.category == 2:
                            # Trap: three squares backward
                            end_state = consequences[i].end_state
                            for _ in range(3):
                                end_state = end_state.predecessor
                            trap_consequence = Consequence(1, consequences[i].probability, end_state)
                        elif consequences[i].end_state.category == 3:
                            # Prison: waste a turn
                            trap_consequence = Consequence(2, consequences[i].probability, consequences[i].end_state)
                        elif consequences[i].end_state.category == 4:
                            # Bonus: free turn!
                            trap_consequence = Consequence(0, consequences[i].probability, consequences[i].end_state)

                        if Action.trigger_probability[action] == 1.0:
                            # Trap trigger for sure.
                            consequences[i] = trap_consequence
                        else:
                            # Apply trigger likelihood
                            consequences[i].probability *= (1-Action.trigger_probability[action])
                            trap_consequence.probability *= Action.trigger_probability[action]
                            trap_results.append(trap_consequence)

                state.results_of[action] = consequences + trap_results

    return states


def markovDecision(layout, circle):
    """ 
    Determines the optimal strategy of a Snakes and Ladders game.
    -------------------------------------------------------------
    Inputs :
        - layout : numpy ndarray. 
                   layout of the traps on the board
        - circle : boolean.
                   if True, you have to land exactly on square 15 to win, 
                   otherwise you restart from the beginning.
    -------------------------------------------------------------
    Outputs :
        - Expec : numpy ndarray. 
                  expected number of turns if optimal strategy is used.
        - Dice  : numpy ndarray.
                  optimal strategy for each square given the layout.
    """
    states = build_states(layout, circle)           # Build the States data structures with associated Consequences
    n_states = len(states)
    expect = [14.0 - i for i in range(n_states)]    # Basic first approximation of the cost
    last_expect = expect[:]
    best_action = [None] * n_states

    max_epoch = 100

    for _ in range(max_epoch):
        for state in states:
            if state.position == 15:
                continue

            best_cost = sys.float_info.max

            for action in Action.names:
                cost = 0
                for conseq in state.results_of[action]: # Extract every generated Consequence and evaluate their costs
                    cost += conseq.probability * (conseq.cost + expect[conseq.end_state.position - 1])

                if cost < best_cost:                    # Store the best action base on its approximation of the expected cost
                    best_cost = cost
                    best_action[state.position - 1] = action

            expect[state.position - 1] = best_cost

        max_delta = max([abs(o - n) for o, n in zip(expect, last_expect)])
        last_expect = expect[:]
        if max_delta < 1e-6:                            # Convergence check
            break

    return np.array(expect[:-1]), np.array(best_action[:-1])


def simulate(layout, looping, actions, n_it):
    states = build_states(layout, looping)
    avg = [0.0] * len(states)
    steps = [[] for _ in range(len(states))]

    for _ in range(n_it):
        for i in range(len(states)):
            state = states[i]
            step = 0
            while True:
                if state.position == 15:
                    avg[i] += float(step) / n_it
                    steps[i].append(float(step))
                    break

                val = random.uniform(0, 1)
                cum_sum = 0.0
                for c in state.results_of[actions[state.position-1]]:
                    cum_sum += c.probability
                    if cum_sum >= val:
                        step += c.cost
                        state = c.end_state
                        break
    return np.array(steps)


#********************************#
#*      Utility Fonctions       *#
#********************************#

def format_5p(e):
    # Format each element of input array to 5 characters.
    result = ""
    for element in e:
        if isinstance(element, float):
            result += f"{element:5.2f} "
        elif isinstance(element, int):
            result += f"{element:5d} "
        else:
            s = element.__str__()
            result += f"{s:5s} "

    return result


def print_run(layout, looping):
    expect, best_actions = markovDecision(layout, looping)
    print("Square " + format_5p(range(1, 16)))
    print("Layout " + format_5p(layout))
    print("Dice   " + format_5p(best_actions))
    print("Expect " + format_5p(expect))

    sim = simulate(layout, looping, best_actions, 10000)
    avg = [np.mean(sim[i]) for i in range(len(sim))]
    print("Simul  " + format_5p(avg))

def store_run(layout, looping, scenario):
    _, best_actions = markovDecision(layout, looping)
    sim = simulate(layout, looping, best_actions, 10000)

    with open("simulation.csv", 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write columns names
        names = [f"state {i}" for i in range(1,16)] + ["scenario"]
        writer.writerow(names)

        for i in range(len(sim[0])):
            row = np.ndarray.tolist(sim[:, i]) + [scenario]
            writer.writerow(row)

def write_csv(data, scenario, looping):
    with open("simulation.csv", 'a') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write columns names
        for i in range(len(data)):
            row = [data[i], scenario, looping]
            writer.writerow(row)

def measurement(store):
    ladder = [0,1,2,2,0,3,0,4,3,2,0,1,1,1,0]        # randomly generated ladder
    
    # Optimal strategy
    _, opti_noLoop = markovDecision(ladder, False)
    sim_opti_noLoop = simulate(ladder, False, opti_noLoop, 1000)
    if store: write_csv(sim_opti_noLoop[0], "Optimal", False)
    print("Optimal noLoop")
    print_run(ladder, False)

    _, opti_loop = markovDecision(ladder, True)
    sim_opti_loop = simulate(ladder, True, opti_loop, 1000)
    if store: write_csv(sim_opti_loop[0], "Optimal", True)
    print("Optimal Loop")
    print_run(ladder, True)
    
    
    # Only secu
    secu = ["secu" for _ in range(len(ladder))]
    sim_secu_noLoop = simulate(ladder, False, secu, 1000)
    if store: write_csv(sim_secu_noLoop[0], "Security", False)

    sim_secu_loop = simulate(ladder, True, secu, 1000)
    if store: write_csv(sim_opti_loop[0], "Security", True)

    # Only normal
    normal = ["normal" for _ in range(len(ladder))]
    sim_normal_noLoop = simulate(ladder, False, normal, 1000)
    if store: write_csv(sim_normal_noLoop[0], "Normal", False)

    sim_normal_loop = simulate(ladder, True, normal, 1000)
    if store: write_csv(sim_normal_loop[0], "Normal", True)
     
    # Random
    rand = [Action.names[random.randint(0,2)] for _ in range(len(ladder))]
    sim_random_noLoop = simulate(ladder, False, rand, 1000)
    if store: write_csv(sim_random_noLoop[0], "Random", False)

    sim_random_loop = simulate(ladder, True, rand, 1000)
    if store: write_csv(sim_random_loop[0], "Random", True)
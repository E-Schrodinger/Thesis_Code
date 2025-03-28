# main.py

import pickle


from Environments.IRP import IRP
from Environments.PD import PD

from Agents.qlearning import Q_Learning
from Agents.qlearning_wsls import Q_Learning_WSLS
from Agents.qlearning_tft import Q_Learning_TfT
from Agents.qlearning_gt import Q_Learning_GT
from Agents.qlearning_self import Q_Learning_Self

from Agents.dec_qlearning import Dec_Q
from Agents.dec_qlearning_wsls import Dec_Q_WSLS
from Agents.dec_qlearning_tft import Dec_Q_Tft
from Agents.dec_qlearning_gt import Dec_Q_GT
from Agents.dec_qlearning_self import Dec_Q_Self

# from Metrics.simulations import Simulations
from Metrics.Simulation_Base import Simulations
from Metrics.Pricing_Metrics import average_price, profit_graph, make_adjacency, state_heatmap, plot_rp, average_price_and_profit
from Metrics.Regret_Metric import find_regret
from Metrics.pd_metrics import action_graph, action_combination_graph

import numpy as np
import os
import time

# Initialize the game environment
game = PD(tmax=6000000, tstable=100000)
print(f'Max val = {game.tmax}')

# p1,p2 = game.compute_p_competitive_monopoly()
# print(p1)
# print(p2)

d_action_space = [0,1]

time_step = 1000

exploration = 0.2
delta_val = 0.95

# Initialize agents

# Agent1 = Q_Learning(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)
# Agent2 = Q_Learning(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)

# Agent1 = Q_Learning_WSLS(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)
# Agent2 = Q_Learning_WSLS(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)

# Agent1 = Q_Learning_TfT(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)
# Agent2 = Q_Learning_TfT(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)

# Agent1 = Q_Learning_GT(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)
# Agent2 = Q_Learning_GT(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)

# Agent1 = Q_Learning_Self(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)
# Agent2 = Q_Learning_Self(game, beta=0.00001, Qinit='uniform', a1_prices = d_action_space, delta = delta_val)


# Agent1 = Dec_Q(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)
# Agent2 = Dec_Q(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)

# Agent1 = Dec_Q_WSLS(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)
# Agent2 = Dec_Q_WSLS(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)

# Agent1 = Dec_Q_GT(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)
# Agent2 = Dec_Q_GT(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)

# Agent1 = Dec_Q_Tft(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)
# Agent2 = Dec_Q_Tft(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)

# Agent1 = Dec_Q_Tft(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)
# Agent2 = Dec_Q_Tft(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)

Agent1 = Dec_Q_Self(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)
Agent2 = Dec_Q_Self(game, batch_size = 200000, Qinit='uniform', a1_prices = d_action_space, pr_explore = exploration, delta = delta_val)

start_time = time.time()

# Initialize the Simulations class with the desired number of iterations

SIMULATION_ITERATIONS = 50 # Example: 100 iterations

SM = Simulations(game, Agent1, Agent2, iterations=SIMULATION_ITERATIONS, ts = time_step, save_agents = True)

print(Agent1.a1_prices)
a1_lists, a2_lists, Agent1_lists, Agent2_lists = SM.get_values()

print(time.time()-start_time)

def save_simulation_data(a1_lists, a2_lists, Agent1lists, Agent2lists, filename="simulation_data.pkl"):
    data = {
        'a1_lists': a1_lists,
        'a2_lists': a2_lists,
        'Agent1lists': Agent1lists,
        'Agent2lists': Agent2lists
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

save_simulation_data(a1_lists, a2_lists, Agent1_lists, Agent2_lists, filename = "DecQself")
action_graph(a1_lists, a2_lists, window_size=100)

print(Agent1.Q)
print(Agent2.Q)

action_combination_graph(a1_lists, a2_lists, window_size=100)
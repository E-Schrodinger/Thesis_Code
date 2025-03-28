import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from Metrics.pd_metrics import*


# List of files to load
filenames = ["data/Q", "data/QWSLS", "data/QGT", "data/QTfT"]

# Optional custom labels for each file
labels = ["Full", "WSLS", "GT", "TfT"]

# Generate the plot
# results = multi_file_a1_action_graph(filenames, window_size=100, labels=labels)

a1_lists, a2_lists, Agent1_lists, Agent2_lists = load_simulation_data(filename="data/Qself")
# results = check_strats(Agent1_lists, Agent2_lists, tft_checker, window_size=1)

graph = action_graph(a1_lists, a2_lists, window_size=100)

action_combination_graph(a1_lists, a2_lists, window_size=100)
# check_joint_equilibria(Agent1_lists, Agent2_lists, tft_checker, window_size=1)




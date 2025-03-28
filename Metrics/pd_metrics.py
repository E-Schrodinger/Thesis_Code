import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import os

def action_graph(a1_lists, a2_lists, window_size=5):
    """
    Plot the rolling average of actions over time for both players.
    For shorter trajectories, extends the final action value to match the longest trajectory.

    Parameters
    ----------
    game : object
        The game environment.
    a1_lists : list of lists
        List where each sublist contains the actions taken by Agent1 in a simulation.
    a2_lists : list of lists
        List where each sublist contains the actions taken by Agent2 in a simulation.
    window_size : int, optional
        The number of past actions to include in the rolling average. Default is 5.

    Returns
    -------
    dict
        A dictionary containing rolling average action statistics over time.
    """
    if len(a1_lists) != len(a2_lists):
        raise ValueError("The number of a1_lists and a2_lists must be the same.")
    
    num_simulations = len(a1_lists)
    max_length = max(max(len(sim) for sim in a1_lists), max(len(sim) for sim in a2_lists))
    
    # Initialize arrays for actions
    player1_actions = np.full((num_simulations, max_length), np.nan)
    player2_actions = np.full((num_simulations, max_length), np.nan)
    
    # Store actions for each simulation, extending final values for shorter trajectories
    for i, (a1_list, a2_list) in enumerate(zip(a1_lists, a2_lists)):
        a1_len = len(a1_list)
        a2_len = len(a2_list)
        
        if a1_len > 0:
            # Fill with actual values up to the end of the trajectory
            player1_actions[i, :a1_len] = a1_list
            # Extend the final value to the end of the max_length
            if a1_len < max_length:
                player1_actions[i, a1_len:] = a1_list[-1]
                
        if a2_len > 0:
            # Fill with actual values up to the end of the trajectory
            player2_actions[i, :a2_len] = a2_list
            # Extend the final value to the end of the max_length
            if a2_len < max_length:
                player2_actions[i, a2_len:] = a2_list[-1]
    
    # Calculate mean actions at each time step across simulations
    mean_player1_actions = np.nanmean(player1_actions, axis=0)
    mean_player2_actions = np.nanmean(player2_actions, axis=0)
    
    # Calculate rolling averages
    rolling_avg_player1 = np.full_like(mean_player1_actions, np.nan)
    rolling_avg_player2 = np.full_like(mean_player2_actions, np.nan)
    
    for i in range(len(mean_player1_actions)):
        start_idx = max(0, i - window_size + 1)
        rolling_avg_player1[i] = np.nanmean(mean_player1_actions[start_idx:i+1])
        rolling_avg_player2[i] = np.nanmean(mean_player2_actions[start_idx:i+1])
    
    time = np.arange(1, max_length + 1)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot rolling average actions
    plt.plot(time, rolling_avg_player1, label=f'Player 1 ({window_size}-period Rolling Avg)')
    plt.plot(time, rolling_avg_player2, label=f'Player 2 ({window_size}-period Rolling Avg)')
    
    plt.ylim(0, 1)
    plt.xlabel('Time')
    plt.ylabel(f'Rolling Average Action (window={window_size})')
    plt.title(f'Rolling Average Action over Time for Both Players')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()
    
    return {
        'time': time,
        'rolling_avg_player1': rolling_avg_player1,
        'rolling_avg_player2': rolling_avg_player2
    }


def action_combination_graph(a1_lists, a2_lists, window_size=5):
    """
    Plot the rolling average proportion of trajectories that played each action combination over time.
    For shorter trajectories, extends the final action combinations to match the longest trajectory.
    
    Parameters
    ----------
    a1_lists : list of lists
        List where each sublist contains the actions (0/1) taken by Agent1 in a simulation.
    a2_lists : list of lists
        List where each sublist contains the actions (0/1) taken by Agent2 in a simulation.
    window_size : int, optional
        The number of past time steps to include in the rolling average. Default is 5.
        
    Returns
    -------
    dict
        A dictionary containing the proportions and rolling averages of each action combination over time.
    """
    if len(a1_lists) != len(a2_lists):
        raise ValueError("The number of a1_lists and a2_lists must be the same.")
    
    num_simulations = len(a1_lists)
    max_length = max(max(len(sim) for sim in a1_lists), max(len(sim) for sim in a2_lists))
    
    # Initialize arrays to count combinations
    count_00 = np.zeros(max_length)  # Both played 0
    count_11 = np.zeros(max_length)  # Both played 1
    count_10 = np.zeros(max_length)  # Player 1 played 1, Player 2 played 0
    count_01 = np.zeros(max_length)  # Player 1 played 0, Player 2 played 1
    valid_trajectories = np.zeros(max_length)  # Count of valid trajectories at each time step
    
    # Count combinations at each time step
    for t in range(max_length):
        for i in range(num_simulations):
            a1_len = len(a1_lists[i])
            a2_len = len(a2_lists[i])
            
            # If both trajectories have at least one action
            if a1_len > 0 and a2_len > 0:
                valid_trajectories[t] += 1
                
                # Get the actions, extending the final value if needed
                a1_idx = min(t, a1_len - 1)  # Use the last action if t is beyond a1_len
                a2_idx = min(t, a2_len - 1)  # Use the last action if t is beyond a2_len
                
                a1 = a1_lists[i][a1_idx]
                a2 = a2_lists[i][a2_idx]
                
                if a1 == 0 and a2 == 0:
                    count_00[t] += 1
                elif a1 == 1 and a2 == 1:
                    count_11[t] += 1
                elif a1 == 1 and a2 == 0:
                    count_10[t] += 1
                elif a1 == 0 and a2 == 1:
                    count_01[t] += 1
    
    # Convert counts to proportions
    prop_00 = np.divide(count_00, valid_trajectories, where=valid_trajectories!=0)
    prop_11 = np.divide(count_11, valid_trajectories, where=valid_trajectories!=0)
    prop_10 = np.divide(count_10, valid_trajectories, where=valid_trajectories!=0)
    prop_01 = np.divide(count_01, valid_trajectories, where=valid_trajectories!=0)
    
    # Calculate rolling averages
    rolling_00 = np.full_like(prop_00, np.nan)
    rolling_11 = np.full_like(prop_11, np.nan)
    rolling_10 = np.full_like(prop_10, np.nan)
    rolling_01 = np.full_like(prop_01, np.nan)
    
    for i in range(len(prop_00)):
        start_idx = max(0, i - window_size + 1)
        rolling_00[i] = np.nanmean(prop_00[start_idx:i+1])
        rolling_11[i] = np.nanmean(prop_11[start_idx:i+1])
        rolling_10[i] = np.nanmean(prop_10[start_idx:i+1])
        rolling_01[i] = np.nanmean(prop_01[start_idx:i+1])
    
    time = np.arange(1, max_length + 1)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot rolling averages
    plt.plot(time, rolling_00, label='0/0 (Both 0)', linestyle='-')
    plt.plot(time, rolling_11, label='1/1 (Both 1)', linestyle='-')
    plt.plot(time, rolling_10, label='1/0 (P1:1, P2:0)', linestyle='-')
    plt.plot(time, rolling_01, label='0/1 (P1:0, P2:1)', linestyle='-')
    
    plt.xlabel('Time')
    plt.ylabel(f'Rolling Avg Proportion (window={window_size})')
    plt.title(f'Rolling Average Proportion of Action Combinations Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits to [0,1] since these are proportions
    plt.ylim(-0.05, 1.05)
    
    plt.show()
    
    return {
        'time': time,
        'proportion_00': prop_00,
        'proportion_11': prop_11,
        'proportion_10': prop_10,
        'proportion_01': prop_01,
        'rolling_avg_00': rolling_00,
        'rolling_avg_11': rolling_11,
        'rolling_avg_10': rolling_10,
        'rolling_avg_01': rolling_01,
        'valid_trajectories': valid_trajectories
    }


def full_checker(Q_val):
    if np.argmax(Q_val[(0,0)]) == 0 and np.argmax(Q_val[(0,1)]) == 0 and np.argmax(Q_val[(1,0)]) == 0 and np.argmax(Q_val[(1,1)]) == 0:
        strat = "All D"
    elif np.argmax(Q_val[(0,0)]) == 1 and np.argmax(Q_val[(0,1)]) == 0 and np.argmax(Q_val[(1,0)]) == 0 and np.argmax(Q_val[(1,1)]) == 1:
        strat = "WSLS"
    elif np.argmax(Q_val[(0,0)]) == 0 and np.argmax(Q_val[(0,1)]) == 0 and np.argmax(Q_val[(1,0)]) == 0 and np.argmax(Q_val[(1,1)]) == 1:
        strat = "GT"
    elif np.argmax(Q_val[(0,0)]) == 0 and np.argmax(Q_val[(0,1)]) == 1 and np.argmax(Q_val[(1,0)]) == 0 and np.argmax(Q_val[(1,1)]) == 1: 
        strat = 'TfT'  
    else: 
        strat = "Other"
    return strat


def tft_checker(Q_val):
    if np.argmax(Q_val[0]) == 0 and np.argmax(Q_val[1]) == 1:
        strat = "Tft"
    elif np.argmax(Q_val[0]) == 0 and np.argmax(Q_val[1]) == 0:
        strat = "All D"
    elif np.argmax(Q_val[0]) == 1 and np.argmax(Q_val[1]) == 1:
        strat = "All C"
    else: 
        strat = "Anti-TfT"
    return strat


def gt_checker(Q_val):
    if np.argmax(Q_val[0]) == 0 and np.argmax(Q_val[1]) == 1:
        strat = "GT"
    elif np.argmax(Q_val[0]) == 0 and np.argmax(Q_val[1]) == 0:
        strat = "All D"
    else: 
        strat = "Other"
    return strat

def wsls_checker(Q_val):
    if np.argmax(Q_val[0]) == 0 and np.argmax(Q_val[1]) == 1:
        strat = "WSLS"
    elif np.argmax(Q_val[0]) == 0 and np.argmax(Q_val[1]) == 0:
        strat = "All D"
    else: 
        strat = "Other"
    return strat

def check_strats(Agent1_lists, Agent2_lists, strats_checker, window_size=100):
    """
    Plot the rolling average proportion of trajectories following different strategies
    over time, based on Q values, using a provided strategy checking function.
    
    Parameters
    ----------
    Agent1_lists : list of lists
        List where each sublist contains Agent1 objects in a simulation.
    Agent2_lists : list of lists
        List where each sublist contains Agent2 objects in a simulation.
    strats_checker : function
        A function that takes Q values and returns a strategy label/name.
    window_size : int, optional
        The number of past time steps to include in the rolling average. Default is 100.
        
    Returns
    -------
    dict
        A dictionary containing the proportions and rolling averages of each strategy over time.
    """
    if len(Agent1_lists) != len(Agent2_lists):
        raise ValueError("The number of Agent1_lists and Agent2_lists must be the same.")
    
    num_simulations = len(Agent1_lists)
    max_length = max(max(len(sim) for sim in Agent1_lists), max(len(sim) for sim in Agent2_lists))
    
    # First pass: identify all unique strategy labels
    p1_strategies = set()
    p2_strategies = set()
    
    for i in range(num_simulations):
        for t in range(min(len(Agent1_lists[i]), max_length)):
            strat = strats_checker(Agent1_lists[i][t].Q)
            p1_strategies.add(strat)
        
        for t in range(min(len(Agent2_lists[i]), max_length)):
            strat = strats_checker(Agent2_lists[i][t].Q)
            p2_strategies.add(strat)
    
    # Initialize dictionaries to count and track strategies
    p1_counts = {strat: np.zeros(max_length) for strat in p1_strategies}
    p2_counts = {strat: np.zeros(max_length) for strat in p2_strategies}
    
    valid_trajectories = np.zeros(max_length)  # Count of valid trajectories at each time step
    
    # Count strategies at each time step
    for t in range(max_length):
        for i in range(num_simulations):
            a1_len = len(Agent1_lists[i])
            a2_len = len(Agent2_lists[i])
            
            # If both trajectories have at least one agent
            if a1_len > 0 and a2_len > 0:
                valid_trajectories[t] += 1
                
                # Get the agents, extending the final value if needed
                a1_idx = min(t, a1_len - 1)  # Use the last agent if t is beyond a1_len
                a2_idx = min(t, a2_len - 1)  # Use the last agent if t is beyond a2_len
                
                a1_agent = Agent1_lists[i][a1_idx]
                a2_agent = Agent2_lists[i][a2_idx]
                
                # Get Q values and determine strategies
                a1_strat = strats_checker(a1_agent.Q)
                a2_strat = strats_checker(a2_agent.Q)
                
                # Count strategies
                p1_counts[a1_strat][t] += 1
                p2_counts[a2_strat][t] += 1
    
    # Convert counts to proportions
    p1_props = {strat: np.divide(counts, valid_trajectories, where=valid_trajectories!=0) 
                for strat, counts in p1_counts.items()}
    p2_props = {strat: np.divide(counts, valid_trajectories, where=valid_trajectories!=0) 
                for strat, counts in p2_counts.items()}
    
    # Calculate rolling averages
    p1_rolling = {strat: np.full_like(props, np.nan) for strat, props in p1_props.items()}
    p2_rolling = {strat: np.full_like(props, np.nan) for strat, props in p2_props.items()}
    
    for i in range(max_length):
        start_idx = max(0, i - window_size + 1)
        
        for strat in p1_strategies:
            p1_rolling[strat][i] = np.nanmean(p1_props[strat][start_idx:i+1])
        
        for strat in p2_strategies:
            p2_rolling[strat][i] = np.nanmean(p2_props[strat][start_idx:i+1])
    
    time = np.arange(1, max_length + 1)
    
    # Create two subplots for Player 1 and Player 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Define colors for up to 10 different strategies
    colors = ['blue', 'red', 'green', 'orange', 'purple', 
              'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot Player 1 strategies
    for i, strat in enumerate(sorted(p1_strategies)):
        color = colors[i % len(colors)]
        ax1.plot(time, p1_rolling[strat], label=strat, color=color)
    
    ax1.set_ylabel(f'Rolling Avg Proportion (window={window_size})')
    ax1.set_title(f'Player 1 Strategy Distribution Over Time')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot Player 2 strategies
    for i, strat in enumerate(sorted(p2_strategies)):
        color = colors[i % len(colors)]
        ax2.plot(time, p2_rolling[strat], label=strat, color=color)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel(f'Rolling Avg Proportion (window={window_size})')
    ax2.set_title(f'Player 2 Strategy Distribution Over Time')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    # Prepare return values
    result = {
        'time': time,
        'player1': {
            'proportions': p1_props,
            'rolling_avgs': p1_rolling
        },
        'player2': {
            'proportions': p2_props,
            'rolling_avgs': p2_rolling
        },
        'valid_trajectories': valid_trajectories
    }
    
    return result

def check_joint_equilibria(Agent1_lists, Agent2_lists, strats_checker, window_size=100):
    """
    Plot the rolling average proportion of trajectories where both players follow the same
    strategy (joint equilibrium) over time.
    
    Parameters
    ----------
    Agent1_lists : list of lists
        List where each sublist contains Agent1 objects in a simulation.
    Agent2_lists : list of lists
        List where each sublist contains Agent2 objects in a simulation.
    strats_checker : function
        A function that takes Q values and returns a strategy label/name.
    window_size : int, optional
        The number of past time steps to include in the rolling average. Default is 100.
        
    Returns
    -------
    dict
        A dictionary containing the proportions and rolling averages of each joint equilibrium over time.
    """
    if len(Agent1_lists) != len(Agent2_lists):
        raise ValueError("The number of Agent1_lists and Agent2_lists must be the same.")
    
    num_simulations = len(Agent1_lists)
    max_length = max(max(len(sim) for sim in Agent1_lists), max(len(sim) for sim in Agent2_lists))
    
    # First pass: identify all possible joint equilibrium strategies
    joint_strategies = set()
    
    for i in range(num_simulations):
        for t in range(max_length):
            a1_len = len(Agent1_lists[i])
            a2_len = len(Agent2_lists[i])
            
            if a1_len > 0 and a2_len > 0:
                a1_idx = min(t, a1_len - 1)
                a2_idx = min(t, a2_len - 1)
                
                a1_strat = strats_checker(Agent1_lists[i][a1_idx].Q)
                a2_strat = strats_checker(Agent2_lists[i][a2_idx].Q)
                
                # Only consider cases where both players have the same strategy
                if a1_strat == a2_strat:
                    joint_strategies.add(a1_strat)
    
    # Initialize dictionaries to count and track joint strategies
    joint_counts = {strat: np.zeros(max_length) for strat in joint_strategies}
    no_equilibrium_counts = np.zeros(max_length)  # Count trajectories not in any equilibrium
    
    valid_trajectories = np.zeros(max_length)  # Count of valid trajectories at each time step
    
    # Count joint strategies at each time step
    for t in range(max_length):
        for i in range(num_simulations):
            a1_len = len(Agent1_lists[i])
            a2_len = len(Agent2_lists[i])
            
            # If both trajectories have at least one agent
            if a1_len > 0 and a2_len > 0:
                valid_trajectories[t] += 1
                
                # Get the agents, extending the final value if needed
                a1_idx = min(t, a1_len - 1)
                a2_idx = min(t, a2_len - 1)
                
                a1_agent = Agent1_lists[i][a1_idx]
                a2_agent = Agent2_lists[i][a2_idx]
                
                # Get Q values and determine strategies
                a1_strat = strats_checker(a1_agent.Q)
                a2_strat = strats_checker(a2_agent.Q)
                
                # Only count when both players have the same strategy
                if a1_strat == a2_strat:
                    joint_counts[a1_strat][t] += 1
                else:
                    no_equilibrium_counts[t] += 1
    
    # Convert counts to proportions
    joint_props = {strat: np.divide(counts, valid_trajectories, where=valid_trajectories!=0) 
                  for strat, counts in joint_counts.items()}
    no_equilibrium_props = np.divide(no_equilibrium_counts, valid_trajectories, where=valid_trajectories!=0)
    
    # Calculate rolling averages
    joint_rolling = {strat: np.full_like(props, np.nan) for strat, props in joint_props.items()}
    no_equilibrium_rolling = np.full_like(no_equilibrium_props, np.nan)
    
    for i in range(max_length):
        start_idx = max(0, i - window_size + 1)
        
        for strat in joint_strategies:
            joint_rolling[strat][i] = np.nanmean(joint_props[strat][start_idx:i+1])
        
        no_equilibrium_rolling[i] = np.nanmean(no_equilibrium_props[start_idx:i+1])
    
    time = np.arange(1, max_length + 1)
    
    # Create a single plot for joint equilibria
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for different strategies
    colors = ['blue', 'red', 'green', 'orange', 'purple', 
              'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot joint equilibrium strategies
    for i, strat in enumerate(sorted(joint_strategies)):
        color = colors[i % len(colors)]
        ax.plot(time, joint_rolling[strat], label=f"Joint {strat}", color=color)
    
    # Plot the "no equilibrium" line
    ax.plot(time, no_equilibrium_rolling, label="No Joint Equilibrium", color='black', linestyle='--')
    
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Rolling Avg Proportion (window={window_size})')
    ax.set_title(f'Joint Equilibrium Distribution Over Time')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    # Prepare return values
    result = {
        'time': time,
        'joint_equilibria': {
            'proportions': joint_props,
            'rolling_avgs': joint_rolling
        },
        'no_equilibrium': {
            'proportions': no_equilibrium_props,
            'rolling_avg': no_equilibrium_rolling
        },
        'valid_trajectories': valid_trajectories
    }
    
    return result

def load_simulation_data(filename="simulation_data.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['a1_lists'], data['a2_lists'], data['Agent1lists'], data['Agent2lists']

def multi_file_a1_action_graph(filenames, window_size=5, labels=None):
    """
    Plot the rolling average of a1_lists actions over time from multiple files.
    
    Parameters
    ----------
    filenames : list
        List of filenames to load data from.
    window_size : int, optional
        The number of past actions to include in the rolling average. Default is 5.
    labels : list, optional
        List of labels for each file in the legend. If None, filenames are used.
        
    Returns
    -------
    dict
        A dictionary containing rolling average action statistics over time for each file.
    """
    if labels is None:
        labels = [os.path.basename(f) for f in filenames]
    
    if len(labels) != len(filenames):
        raise ValueError("Number of labels must match number of filenames")
    
    plt.figure(figsize=(12, 7))
    
    results = {}
    
    for i, filename in enumerate(filenames):
        # Load data
        a1_lists, _, _, _ = load_simulation_data(filename=filename)
        
        num_simulations = len(a1_lists)
        max_length = max(len(sim) for sim in a1_lists)
        
        # Initialize arrays for actions
        player1_actions = np.full((num_simulations, max_length), np.nan)
        
        # Store actions for each simulation, extending final values for shorter trajectories
        for j, a1_list in enumerate(a1_lists):
            a1_len = len(a1_list)
            
            if a1_len > 0:
                # Fill with actual values up to the end of the trajectory
                player1_actions[j, :a1_len] = a1_list
                # Extend the final value to the end of the max_length
                if a1_len < max_length:
                    player1_actions[j, a1_len:] = a1_list[-1]
        
        # Calculate mean actions at each time step across simulations
        mean_player1_actions = np.nanmean(player1_actions, axis=0)
        
        # Calculate rolling averages
        rolling_avg_player1 = np.full_like(mean_player1_actions, np.nan)
        
        for j in range(len(mean_player1_actions)):
            start_idx = max(0, j - window_size + 1)
            rolling_avg_player1[j] = np.nanmean(mean_player1_actions[start_idx:j+1])
        
        time = np.arange(1, max_length + 1)
        
        # Plot rolling average actions
        plt.plot(time, rolling_avg_player1, label=f'{labels[i]} ({window_size}-period Rolling Avg)')
        
        # Store results
        results[labels[i]] = {
            'time': time,
            'rolling_avg_player1': rolling_avg_player1
        }
    
    plt.xlabel('Time')
    plt.ylabel(f'Rolling Average Action (window={window_size})')
    plt.title(f'Rolling Average Action over Time for Player 1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()
    
    return results

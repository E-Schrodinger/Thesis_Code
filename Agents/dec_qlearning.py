

import sys
import numpy as np
import copy
from Agents.QBase import QBase

class Dec_Q(QBase):
    """
    A class implementing Decentralized Q Learning for reinforcement learning in games.

    This class provides methods for initializing, updating, and using a Q-function
    with batch updates for SARSA learning in a game-theoretic context.

    Attributes:
    ----------
    delta : float
        Discount factor for future rewards (default: 0.95).
    epsilon : float
        Exploration rate for epsilon-greedy strategy (default: 0.1).
    beta : float
        Decay rate for exploration probability (default: 4e-6).
    batch_size : int
        Number of steps between batch updates (default: 1000).
    Q : ndarray
        Q-function storing action-value estimates.
    Q_val : ndarray
        Copy of Q-function used for value updates.
    trans : ndarray
        Transition function counting state transitions.
    num : ndarray
        Counter for state-action visits.
    reward : ndarray
        Accumulated rewards for each state-action pair.
    X : ndarray
        Probability distribution for action selection.
    """

    def __init__(self, game, **kwargs):
        """
        Initialize the Batch SARSA agent.

        Parameters:
        ----------
        game : object
            The game environment.
        **kwargs : dict
            Additional parameters to override default values.
        """
 
        super().__init__(game, **kwargs)
        
        self.pr_explore = kwargs.get('pr_explore', 0.05)
        self.alpha = kwargs.get('alpha', 0.15)
        self.batch_size = kwargs.get('batch_size', 100)
        self.lamb = kwargs.get('lamb', 0.1)


        


  
    
  
    def reset(self, game):
        """Reset all data structures to initial state"""
        self.price_state_space = copy.copy(self.a1_space)
        self.Q = self.make_Q()
        self.Q_val = self.Q.copy()
        self.num = self.make_num()
        

    def pick_strategies(self, game, p, t):
        """
        Choose actions based on the current Q-function and exploration strategy.

        This method implements an epsilon-greedy strategy with decaying exploration rate.

        Parameters:
        ----------
        game : object
            The game environment.
        s : tuple
            Current state.
        t : int
            Current time step.

        Returns:
        -------
        ndarray
            Chosen actions for each player.
        """
        s = p
        a = np.zeros(1)
        # Calculate exploration probability
   
        
        # Determine whether to explore or exploit for each player
        e = (self.pr_explore > np.random.rand())
        
        if e:
            # Explore: choose a random action
            a = np.random.randint(0, self.k)
        else:
            # Exploit: choose the action with the highest Q-value
            a = np.argmax(self.Q[tuple(s)])
    
        self.a_price = self.a1_space[a]
        return self.a_price
    
    def X_function(self, game, s, a):
        """
        Calculate action selection probabilities.

        Parameters:
        ----------
        game : object
            The game environment.
        s : tuple
            Current state.
        a : int
            Action to calculate probability for.

        Returns:
        -------
        ndarray
            Probabilities of selecting action a for each player.
        """
        
    
        optimal = np.argmax(self.Q_val[tuple(s)])
        if a == optimal:
            probabilities = self.pr_explore/self.Q_val.shape[1] + 1 - self.pr_explore
        else:
            probabilities = self.pr_explore/self.Q_val.shape[1]
        return probabilities
    
    def adaption_phase(self, game, s_hat, a_hat):
        """
        Perform the adaptation phase of the Batch SARSA algorithm.

        Parameters:
        ----------
        game : object
            The game environment.
        s_hat : tuple
            Current state.
        a_hat : tuple
            Chosen actions.
        s_prime : tuple
            Next state.
        """
        state = tuple(s_hat) + (a_hat,)
        self.Q[state] = self.Q_val[state].copy()

    def update_function(self, game, p, a_prices, pi, stable, t, tol=1e-6):
        """
        Update the Q-function based on the observed transition and reward.

        Parameters:
        ----------
        game : object
            The game environment.
        s : tuple
            Current state.
        a : tuple
            Chosen actions.
        s1 : tuple
            Next state.
        pi : ndarray
            Observed payoffs.
        stable : int
            Number of consecutive stable updates.
        t : int
            Current time step.
        tol : float, optional
            Tolerance for considering Q-values as converged (default: 1e-1).

        Returns:
        -------
        tuple
            Updated Q-function and stability counter.
        """
        self.dt = t
        s = p
        s1 = a_prices
        a = a_prices[0]
       
        subj_state = tuple(s) + (a,)
        # print(subj_state)
        # print(self.Q_val.shape)
        old_value = self.Q_val[subj_state]
        
        # Update counters and accumulated rewards
        self.num[tuple(s)] += 1

        # Calculate learning rate
        a_t = 1/(self.num[tuple(s)]+1)
        
        # Calculate expected Q-value of next state
        Q_merge = 0
        for i in range(self.Q_val.shape[2]):
            Q_merge += self.X_function(game, s1, i) * self.Q_val[ tuple(s1) + (i,)]

        # Update Q-value
        self.Q_val[subj_state] = (1-a_t)*old_value + a_t*(pi + self.delta*Q_merge)

            

        # Perform batch update if necessary
        if (t % self.batch_size == 0):
            old_q = self.Q.copy()
            U = np.random.uniform()
            for s_hat in np.ndindex((self.Q_val.shape[0], self.Q_val.shape[1])):
                for a_hat in np.ndindex(self.Q_val.shape[2]):
                    if U >= self.lamb:
                        self.adaption_phase(game, s_hat, a_hat)
            self.Q_val = self.Q.copy()
            same_q = np.allclose(old_q, self.Q, tol)
            if same_q:
                stable += 1
            else: stable = 0
            self.num.fill(0)
 
       
        return self.Q, stable 
"""
Model of algorithms and competition
"""

import numpy as np
from itertools import product
from scipy.optimize import fsolve
import sys
import copy


class IRP(object):
    """
    IRP Model

    Attributes
    ----------
    n : int
        Number of players.
    alpha : float
        Product differentiation parameter.
    mu : float
        Product differentiation parameter.
    a : int
        Value of the products.
    a0 : float
        Value of the outside option.
    c : float
        Marginal cost.
    tstable : int
        Periods of game stability.
    tmax : int
        Maximum iterations of play.
    dem_function : str or callable
        Demand function to use ('default' or custom function).
    """

    def __init__(self, dem_function='default', **kwargs):
        """
        Initialize the IRP model with given or default parameters.

        Parameters
        ----------
        dem_function : str or callable, optional
            Demand function to use (default is 'default').
        **kwargs : dict
            Additional parameters to override default values.
        """
        # Default properties
        self.n = kwargs.get('n', 2)
        self.c = kwargs.get('c', 1)
        self.a = kwargs.get('a', 2)
        self.a0 = kwargs.get('a0', 0)
        self.mu = kwargs.get('mu', 0.25)
        self.tstable = kwargs.get('tstable', 1e2)
        self.tmax = kwargs.get('tmax', 1e4)

        self.dem_function = dem_function

        # Derived properties
        # self.sdim, self.s0 = self.init_state()
        # self.p_minmax = self.compute_p_competitive_monopoly()
        # self.A = self.init_actions()
        # self.PI = self.init_PI()

    def demand(self, p):
        """
        Compute the demand for each firm given their prices.

        Parameters
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns
        -------
        ndarray
            Array of demand quantities for each firm.
        """
        e = np.exp((self.a - p) / self.mu)
        d = e / (np.sum(e) + np.exp(self.a0 / self.mu))
        return d

    def foc(self, p):
        """
        Compute the first-order condition for profit maximization.

        Parameters
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns
        -------
        ndarray
            Array of first-order condition values.
        """
        if self.dem_function == 'default':
            d = self.demand(p)
        else:
            d = self.dem_function(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu
        return np.squeeze(zero)

    def foc_monopoly(self, p):
        """
        Compute the first-order condition for a monopolist.

        Parameters
        ----------
        p : ndarray
            Array of prices set by the monopolist for each product.

        Returns
        -------
        ndarray
            Array of first-order condition values for a monopolist.
        """
        if self.dem_function == 'default':
            d = self.demand(p)
        else:
            d = self.dem_function(p)
        d1 = np.flip(d)
        p1 = np.flip(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu + (p1 - self.c) * d1 / self.mu
        return np.squeeze(zero)

    def compute_p_competitive_monopoly(self):
        """
        Compute competitive and monopoly prices.

        Returns
        -------
        tuple
            A tuple containing competitive and monopoly prices.
        """
        p0 = np.ones((1, self.n)) * 3 * self.c
        p_competitive = fsolve(self.foc, p0)
        p_monopoly = fsolve(self.foc_monopoly, p0)
        return p_competitive, p_monopoly

    def init_actions(self):
        """
        Initialize the discrete action space (possible prices).

        Returns
        -------
        ndarray
            Array of possible prices.
        """
        a = np.linspace(min(self.p_minmax[0]), max(self.p_minmax[1]), self.k - 2)
        delta = a[1] - a[0]
        A = np.linspace(min(a) - delta, max(a) + delta, self.k)
        return A

    def init_state(self):
        """
        Initialize the state space dimensions and initial state.

        Returns
        -------
        tuple
            A tuple containing state space dimensions and initial state.
        """
        sdim = (self.k, self.k)
        s0 = np.zeros(len(sdim)).astype(int)
        return sdim, s0

    def compute_profits(self, p):
        """
        Compute profits for each firm given their prices.

        Parameters
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns
        -------
        ndarray
            Array of profits for each firm.
        """ 
        d = self.demand(p)
        pi = (p - self.c) * d
        return pi

    def init_PI(self, game):
        """
        Initialize the profit matrix for all possible states and actions.

        Parameters
        ----------
        game : IRP
            The game environment.

        Returns
        -------
        ndarray
            3D array of profits for all possible states and actions.
        """
        PI = np.zeros(game.sdim + (game.n,))
        for s in product(*[range(i) for i in game.sdim]):
            p = np.asarray(game.A[np.asarray(s)])
            PI[s] = game.compute_profits(p)
        return PI


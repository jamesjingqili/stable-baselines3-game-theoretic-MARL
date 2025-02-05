from os import path
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled



class LQR_Env2(gym.Env):
    """
    Clean implementation of linear quadratic game
    
    1. Stage cost of player i:
        x_t_i^T * Q_i * x_t_i + u_t_i^T * R_i * u_t_i
    where x_t_i and u_t_i are the state and the control input of player i at time t.
    
    2. Dynamics:
        x_{t+1} = Ax_t + B1 u_t_1 + B2 u_t_2
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 0.1
        self.num_players = 2
        self.nx = 4 # each player has 4 states
        self.nu = 2 # each player has 2 control inputs
        self.total_state_dim = self.nx * self.num_players
        self.total_action_dim = self.nu * self.num_players
        self.players_u_index_list = np.array([[0, 1], [2, 3]])
        
        # There are two ways to define dynamics, 
        # one is to define the dynamics directly in step() function,
        # and the other is to define the following matrices A and B!
        
        # We start with defining the individual dynamics of each player!
        self.A_individual = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.B_individual = np.array([
            [0, 0],
            [0, 0],
            [self.dt, 0],
            [0, self.dt]
        ])
        # self.A_individual = np.array([1])
        # self.B_individual = np.array([1])
        self.A = np.zeros((self.total_state_dim, self.total_state_dim))
        self.B = np.zeros((self.total_state_dim, self.total_action_dim))
        self.B_list = []
        for i in range(self.num_players):
            self.A[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx] = self.A_individual
            self.B[i*self.nx:(i+1)*self.nx, i*self.nu:(i+1)*self.nu] = self.B_individual
            self.B_list.append(self.B[:, i*self.nu:(i+1)*self.nu])

        # There are two ways to define costs, 
        # one is to define the cost function directly in step() function, 
        # and the other is to define the following cost matrices Q and R! 
        
        # defining the cost function, Q and are R are 3-dimensional arrays, 
        # the third dimension is the player index!
        self.Q = np.zeros((self.total_state_dim, self.total_state_dim, self.num_players))
        self.R = np.zeros((self.total_action_dim, self.total_action_dim, self.num_players))
        state_cost0 = np.zeros((8, 8))
        state_cost0[4:6, 4:6] = np.eye(2)
        state_cost1 = np.zeros((8,8))
        state_cost1[0:2, 0:2] = np.eye(2)
        state_cost1[4:6, 4:6] = np.eye(2)
        state_cost1[0:2, 4:6] = -np.eye(2)
        state_cost1[4:6, 0:2] = -np.eye(2)
        
        for i in range(self.num_players):
            if i == 0:
                # import pdb; pdb.set_trace()
                # self.Q[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx, i] = state_cost0
                self.Q[:, :, i] = state_cost0
                self.R[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu, i] = np.eye(self.nu)
            elif i == 1:
                # self.Q[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx, i] = state_cost1
                self.Q[:, :, i] = state_cost1
                self.R[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu, i] = np.eye(self.nu)
        self.Q_list = [self.Q[:, :, i] for i in range(self.num_players)]
        self.R_list = [self.R[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu, i] for i in range(self.num_players)]
        # defining state and action spaces:
        state_bound = 4
        action_bound = 4
        self.action_space_dim = self.nu * self.num_players
        self.state_space_dim = self.nx * self.num_players
        self.state_high = np.ones((self.state_space_dim,)) * state_bound
        self.state_low = -self.state_high
        self.action_high = np.ones((self.action_space_dim,)) * action_bound
        self.action_low = -self.action_high
        
        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=self.state_low, high=self.state_high, dtype=np.float64
        )
        
        self.is_nonlinear_game = False
    # torch version of the dynamics and cost functions:
    def f_list_torch(self, x, u):
        x_next = torch.tensor(self.A) @ x + torch.tensor(self.B) @ u
        return x_next
    def l_list_torch(self, x, u):
        cost_values = [
            (1/2 * x.T @ torch.tensor(self.Q_list[i]) @ x + 1/2 * u.T @ torch.tensor(self.R_list[i]) @ u + 10).item()/10
            for i in range(self.num_players)
        ]
        return cost_values
    def step(self, u):
        # assert self.action_space.contains(u), f"action {u} is out of action space {self.action_space}"
        # we update state and cost, and return them as the first and second output of step() function
        # update state for each player
        X = self.state.reshape(self.total_state_dim, 1)
        U = u.reshape(self.total_action_dim, 1)
        # U[1] = -X[1]
        next_X = self.A @ X + self.B @ U      
        # update cost for each player
        self.costs = (-np.array([
            (1/2 * X.T @ self.Q[:,:,i] @ X + # @ self.Q[:,:,i] 
            1/2 * U.T  @ self.R[:,:,i] @ U).item() 
            # (np.linalg.norm(X, 1) + np.linalg.norm(U, 1)).item()
            for i in range(self.num_players) # @ self.R[:,:,i] 
        ]) + 10) / 10
        
        
        sum_costs = np.sum(self.costs)
        
        
        # tmp = np.sum(self.costs)
        # self.costs = np.array([tmp for i in range(self.num_players)])

        # check if the state is out of bound:
        # for distinguishing terminated and truncated, 
        # see https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
        terminated = False # check whether state is out of bound
        truncated = False # we don't use here
        self.state = next_X.reshape(self.total_state_dim,)
        if np.any(self.state > self.state_high) or np.any(self.state < self.state_low):
            terminated = True
        info = {"individual_cost": self.costs }#np.array([sum_costs, sum_costs])}
        # note that we set reward, as shown in the second output, 
        # to the sum of players' costs, but we store individual costs in info!
        return self.state, sum_costs, terminated, truncated, info




    # NOTE! we can assign initial state as env.reset(options = {"initial_state": initial_state}) !!!!!
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        initial_state_high = np.array([
            2, 2, 0.2, 0.2, 2, 2, 0.2, 0.2
        ])
        initial_state_low = np.array([
            -2, -2, -0.2, -0.2, -2, -2, -0.2, -0.2
        ])
        # initial_state_high = np.array([1., 1.])
        # initial_state_low = np.array([-1., -1.])
        # initial_state_high = self.state_high
        # initial_state_low = self.state_low
        if options is None:
            self.state = self.np_random.uniform(low=initial_state_low, high=initial_state_high)
        elif "initial_state" in options:
            self.state = options["initial_state"]
        self.costs = np.zeros(self.num_players)
        return self.state, {}

    def render(self):
        pass



class LQR_Env3(gym.Env):
    """
    Clean implementation of linear quadratic game
    
    1. Stage cost of player i:
        x_t_i^T * Q_i * x_t_i + u_t_i^T * R_i * u_t_i
    where x_t_i and u_t_i are the state and the control input of player i at time t.
    
    2. Dynamics:
        x_{t+1} = Ax_t + B1 u_t_1 + B2 u_t_2
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 0.1
        self.num_players = 2
        self.nx = 4 # each player has 4 states
        self.nu = 2 # each player has 2 control inputs
        self.total_state_dim = self.nx * self.num_players
        self.total_action_dim = self.nu * self.num_players
        self.players_u_index_list = np.array([[0, 1], [2, 3]])
        
        # There are two ways to define dynamics, 
        # one is to define the dynamics directly in step() function,
        # and the other is to define the following matrices A and B!
        
        # We start with defining the individual dynamics of each player!
        self.A_individual = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.B_individual = np.array([
            [0, 0],
            [0, 0],
            [self.dt, 0],
            [0, self.dt]
        ])
        # self.A_individual = np.array([1])
        # self.B_individual = np.array([1])
        self.A = np.zeros((self.total_state_dim, self.total_state_dim))
        self.B = np.zeros((self.total_state_dim, self.total_action_dim))
        self.B_list = []
        for i in range(self.num_players):
            self.A[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx] = self.A_individual
            self.B[i*self.nx:(i+1)*self.nx, i*self.nu:(i+1)*self.nu] = self.B_individual
            self.B_list.append(self.B[:, i*self.nu:(i+1)*self.nu])

        # There are two ways to define costs, 
        # one is to define the cost function directly in step() function, 
        # and the other is to define the following cost matrices Q and R! 
        
        # defining the cost function, Q and are R are 3-dimensional arrays, 
        # the third dimension is the player index!
        self.Q = np.zeros((self.total_state_dim, self.total_state_dim, self.num_players))
        self.R = np.zeros((self.total_action_dim, self.total_action_dim, self.num_players))
        state_cost0 = np.zeros((8, 8))
        state_cost0[4, 4] = 1#np.eye(2)
        state_cost1 = np.zeros((8,8))
        state_cost1[0, 0] = 1# np.eye(2)
        state_cost1[4, 4] = 1# np.eye(2)
        state_cost1[0, 4] = -1# np.eye(2)
        state_cost1[4, 0] = -1# np.eye(2)
        
        for i in range(self.num_players):
            if i == 0:
                # import pdb; pdb.set_trace()
                # self.Q[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx, i] = state_cost0
                self.Q[:, :, i] = state_cost0
                self.R[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu, i] = np.eye(self.nu)
            elif i == 1:
                # self.Q[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx, i] = state_cost1
                self.Q[:, :, i] = state_cost1
                self.R[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu, i] = np.eye(self.nu)
        self.Q_list = [self.Q[:, :, i] for i in range(self.num_players)]
        self.R_list = [self.R[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu, i] for i in range(self.num_players)]
        # defining state and action spaces:
        state_bound = 4
        action_bound = 4
        self.action_space_dim = self.nu * self.num_players
        self.state_space_dim = self.nx * self.num_players
        self.state_high = np.ones((self.state_space_dim,)) * state_bound
        self.state_low = -self.state_high
        self.action_high = np.ones((self.action_space_dim,)) * action_bound
        self.action_low = -self.action_high
        
        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=self.state_low, high=self.state_high, dtype=np.float64
        )
        
        self.is_nonlinear_game = False
    # torch version of the dynamics and cost functions:
    def f_list_torch(self, x, u):
        x_next = torch.tensor(self.A) @ x + torch.tensor(self.B) @ u
        return x_next
    def l_list_torch(self, x, u):
        cost_values = [
            (1/2 * x.T @ torch.tensor(self.Q_list[i]) @ x + 1/2 * u.T @ torch.tensor(self.R_list[i]) @ u + 10).item()/10
            for i in range(self.num_players)
        ]
        return cost_values
    def step(self, u):
        # assert self.action_space.contains(u), f"action {u} is out of action space {self.action_space}"
        # we update state and cost, and return them as the first and second output of step() function
        # update state for each player
        X = self.state.reshape(self.total_state_dim, 1)
        U = u.reshape(self.total_action_dim, 1)
        # U[1] = -X[1]
        next_X = self.A @ X + self.B @ U      
        # update cost for each player
        self.costs = (-np.array([
            (1/2 * X.T @ self.Q[:,:,i] @ X + # @ self.Q[:,:,i] 
            1/2 * U.T  @ self.R[:,:,i] @ U).item() 
            # (np.linalg.norm(X, 1) + np.linalg.norm(U, 1)).item()
            for i in range(self.num_players) # @ self.R[:,:,i] 
        ]) + 10) / 10
        
        
        sum_costs = np.sum(self.costs)
        
        terminated = False # check whether state is out of bound
        truncated = False # we don't use here
        self.state = next_X.reshape(self.total_state_dim,)
        # import pdb ; pdb.set_trace()
        if np.any(self.state[[0,2,3,4,6,7]] > self.state_high[[0,2,3,4,6,7]]) or np.any(self.state[[0,2,3,4,6,7]] < self.state_low[[0,2,3,4,6,7]]):
            terminated = True
        info = {"individual_cost": self.costs }#np.array([sum_costs, sum_costs])}
        # note that we set reward, as shown in the second output, 
        # to the sum of players' costs, but we store individual costs in info!
        return self.state, sum_costs, terminated, truncated, info




    # NOTE! we can assign initial state as env.reset(options = {"initial_state": initial_state}) !!!!!
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        initial_state_high = np.array([
            2, 15, 0.2, 0.2, 2, 15, 0.2, 0.2
        ])
        initial_state_low = np.array([
            -2, 0, -0.2, -0.2, -2, 0, -0.2, -0.2
        ])
        if options is None:
            self.state = self.np_random.uniform(low=initial_state_low, high=initial_state_high)
        elif "initial_state" in options:
            self.state = options["initial_state"]
        self.costs = np.zeros(self.num_players)
        return self.state, {}

    def render(self):
        pass

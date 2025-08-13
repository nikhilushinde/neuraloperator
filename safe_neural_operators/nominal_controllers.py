"""
File to define different nominal controllers and disturbance functions for baseline experiments
"""

import torch
import numpy as np
import scipy
from abc import ABC, abstractmethod

####################### 3. Nominal Controller and Disturbance #######################

####################### Controllers #######################


def dummy_zero_controller(x):
    return torch.zeros(2)


class NominalController:
    def __init__(self, system, system_type, min_control, max_control, init_goal_position, verbose=False):
        self.system = system
        supported_systems = ["Quad2DAttitude", "Quad10D"]
        assert system_type in supported_systems, (
            f"System type {system_type} not supported. Supported systems: {supported_systems}"
        )

        # Setup LQR controller
        new_goal_frequency = 0.0001  # Hz (how often to generate new goal)
        if system_type == "Quad2DAttitude":
            self.min_control = np.array([-0.2, 6.0])
            self.max_control = np.array([0.2, 13.0])
        elif system_type == "Quad10D":
            self.min_control = np.array([-0.2, -0.2, 6.0])
            self.max_control = np.array([0.2, 0.2, 13.0])

        self.init_goal_position = init_goal_position  # np.array([0.0, 1.0, 0, 0])
        self.goal = self.init_goal_position

        self.system_type = system_type
        self.init_system(system_type=system_type)
        self.verbose = verbose
        return

    def init_system(self, system_type="Quad2DAttitude"):
        if self.system_type == "Quad2DAttitude":
            # Setup Dynamics Matrices
            # State: [y, z, v_y, v_z]
            # Dynamics:
            # \dot y = v_y + d_1
            # \dot z = v_z + d_2
            # \dot v_y = g * u_1 + d_3
            # \dot v_z = u_2 - g + d_4
            A = np.zeros((4, 4))
            B = np.zeros((4, 2))

            A[0, 2] = 1.0
            A[1, 3] = 1.0

            gravity = self.system.gravity  # 9.81
            B[2, 0] = gravity
            B[3, 1] = 1.0

            self.u_equilibrium = torch.tensor(
                [0, gravity]
            )  # As the system is Ax + Bu + C: need a nominal hover control for the equilibria

            # Set gain matrix
            Q = np.eye(4)
            R = np.eye(2)

            # Old Q matrix 
            # Q[0, 0] = 5.0
            # Q[1, 1] = 5.0
            # Q[2, 2] = 2.0
            # Q[3, 3] = 2.0

            Q[0, 0] = 5.0
            Q[1, 1] = 5.0
            Q[2, 2] = 4.0
            Q[3, 3] = 4.0

            R[0, 0] = 0.1
            R[1, 1] = 0.1
            # gain_matrix = np.zeros((2, 4))
            # gain_matrix[0, 0] = 0.2  # x -> pitch
            # gain_matrix[0, 2] = 0.2  # v_x -> pitch
            # gain_matrix[1, 1] = 10.0  # y -> roll
            # gain_matrix[1, 3] = 10.0  # v_y -> roll
            # self.gain_matrix = torch.from_numpy(-gain_matrix)
            self.gain_matrix = torch.from_numpy(self.create_gain_matrix(A=A, B=B, Q=Q, R=R))
            # self.gain_matrix = self.gain_matrix.to(torch.float32)
        elif self.system_type == "Quad10D":
            gravity = self.system.gravity  # 9.81
            d0 = self.system.d0  # 10.0
            d1 = self.system.d1  # 8.0
            n0 = self.system.n0  # 10.0
            k_T = self.system.k_T  # 0.9
            mass = self.system.mass  # 1.0

            # Setup Dynamics Matrices
            A = np.zeros((10, 10))
            A[0, 1] = 1.0
            A[1, 2] = gravity  # use g * theta_x instead of g * tan(theta_x)
            A[2, 2] = -d1
            A[2, 3] = 1.0
            A[3, 3] = -d0
            A[4, 5] = 1.0
            A[5, 6] = gravity
            A[6, 6] = -d1
            A[6, 7] = 1.0
            A[7, 7] = -d0
            A[8, 9] = 1.0

            B = np.zeros((10, 3))
            B[3, 0] = n0
            B[7, 1] = n0
            B[9, 2] = k_T / mass

            self.u_equilibrium = torch.tensor(
                [0, 0, gravity / k_T]
            )  # As the system is Ax + Bu + C: need a nominal hover control for the equilibria

            # Set gain matrix
            Q = np.eye(10)
            R = np.eye(3)

            # Positional
            Q[0, 0] = 5.0
            Q[4, 4] = 5.0
            Q[8, 8] = 8.0
            # Velocity
            Q[1, 1] = 2.0
            Q[5, 5] = 2.0
            Q[9, 9] = 2.0
            # Angles
            Q[2, 2] = 3.0
            Q[6, 6] = 3.0
            # Angular Velocity
            Q[3, 3] = 1.0
            Q[7, 7] = 1.0

            R = R * 0.1

            self.gain_matrix = torch.from_numpy(self.create_gain_matrix(A=A, B=B, Q=Q, R=R))

            # gain_matrix = np.zeros((3, 10))
            # gain_matrix[0, 0] = 0.2  # x -> pitch
            # gain_matrix[0, 1] = 0.2  # v_x -> pitch
            # gain_matrix[1, 4] = 0.2  # y -> roll
            # gain_matrix[1, 5] = 0.2  # v_y -> roll
            # gain_matrix[2, 8] = 10.0  # z -> thrust
            # gain_matrix[2, 9] = 10.0  # v_z -> thrust
            # self.gain_matrix = torch.from_numpy(-gain_matrix)
            self.gain_matrix = self.gain_matrix.to(torch.float32)
        return

    def create_gain_matrix(self, A, B, Q, R):
        """
        Solve for the LQR gain matrix
        """
        N = 50

        # Solve for P with the discrete-time algebraic Riccati equation using dynamic programming
        P = [None] * (N + 1)
        P[N] = Q
        for i in range(N, 0, -1):
            P[i - 1] = Q + A.T @ P[i] @ A - A.T @ P[i] @ B @ np.linalg.inv(R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

        # https://www.mwm.im/lqr-controllers-with-python/
        # P_new = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        # K_new = -1 * np.linalg.inv(R + B.T @ P_new @ B) @ (B.T @ P_new @ A)
        # # Calculate the optimal feedback gain matrix using the converged P
        # # K = -1 * np.linalg.inv(R + B.T @ P[0] @ B) @ (B.T @ P[0] @ A)
        # K = K_new

        # https://www.mwm.im/lqr-controllers-with-python/
        # first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(R) * (B.T * X))
        if self.verbose:
            print("Gain matrix: ", K)

        # import pdb; pdb.set_trace()

        return -K

    def set_goal(self, goal):
        self.goal = goal
        return

    def get_goal(self):
        return self.goal

    def __call__(self, state):
        # goal_full = np.concatenate([self.goal_position, np.zeros(4)])
        # u_nom = self.u_hover + self.gain_matrix @ (state - goal_full)
        # u_nom = np.clip(u_nom, self.min_control, self.max_control)

        if type(state) == np.ndarray:
            state = torch.from_numpy(state)

        # import pdb; pdb.set_trace()
        u_nom = self.u_equilibrium + self.gain_matrix @ (state - self.goal)
        u_nom = np.clip(u_nom, self.min_control, self.max_control)
        return u_nom


class randomGoalNominalController(NominalController):
    def __init__(
        self,
        system,
        system_type="Quad2DAttitude",
        min_control=[-0.2, 6],
        max_control=[0.2, 13],
        init_goal_position=None,  # [-2.0, 1.5, 0.0, 0.0],
        env=None,
        goal_threshold=0.1,
        goal_reset_step=np.inf,
        verbose=False
    ):
        self.env = env
        self.step_since_goal_reset = 0
        self.goal_threshold = goal_threshold
        self.goal_reset_step = goal_reset_step
        self.verbose = verbose

        if self.env is not None:
            self.plot_state_bounds = env.system.state_test_range()
            self.avoid_fn = env.system.avoid_fn
        else:
            self.plot_state_bounds = [[-5, 5], [-5, 5]]
            self.avoid_fn = lambda x: 1.0  # Positive return doesn't avoid anything

        self.system_type = system_type

        if init_goal_position is None:
            init_goal_position = self.generate_random_goal()
        else:
            self.generate_random_goal()

        super().__init__(
            system,
            system_type=system_type,
            min_control=min_control,
            max_control=max_control,
            init_goal_position=init_goal_position,
            verbose=verbose
        )

        return

    def set_goal(self, goal):
        retval = super().set_goal(goal)
        self.step_since_goal_reset = 0
        return retval

    def generate_random_goal(self):
        new_goal_in_avoid = True

        if self.system_type == "Quad2DAttitude":
            self.x_bounds = self.plot_state_bounds[0]
            self.y_bounds = self.plot_state_bounds[1]
        elif self.system_type == "Quad10D":
            self.x_bounds = self.plot_state_bounds[0]
            self.y_bounds = self.plot_state_bounds[4]
            self.z_bounds = self.plot_state_bounds[8]

        while new_goal_in_avoid:
            if self.system_type == "Quad2DAttitude":
                new_goal = np.array(
                    [
                        np.random.uniform(low=self.x_bounds[0], high=self.x_bounds[1]),
                        np.random.uniform(low=self.y_bounds[0], high=self.y_bounds[1]),
                        # np.random.uniform(low=self.y_bounds[0], high=0.5), # so that goal in the wind
                        0.0,
                        0.0,
                    ]
                )
            elif self.system_type == "Quad10D":
                new_goal = np.zeros(10)
                new_goal[0] = np.random.uniform(low=self.x_bounds[0], high=self.x_bounds[1])
                new_goal[4] = np.random.uniform(low=self.y_bounds[0], high=self.y_bounds[1])
                new_goal[8] = np.random.uniform(low=self.z_bounds[0], high=self.z_bounds[1])

            new_goal_in_avoid = self.avoid_fn(torch.from_numpy(new_goal)) < 0

        if self.verbose: 
            print(f"\n\n New goal: {new_goal}")
        return new_goal

    def __call__(self, state):
        control = super().__call__(state)
        self.step_since_goal_reset += 1

        # If Error is close to zero generate a new goal and update
        if np.linalg.norm(state - self.goal) < self.goal_threshold or self.step_since_goal_reset > self.goal_reset_step:
            new_goal = self.generate_random_goal()
            self.set_goal(new_goal)
            if self.verbose: 
                print(f"\nNew goal: {new_goal}\n")

        return control

    def reset(self):
        """
        Reset the controller to the initial state
        """
        self.set_goal(self.init_goal_position)
        np.random.seed(3)
        self.generate_random_goal()
        return
    
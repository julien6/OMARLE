import copy
import functools
import json
import os
import gymnasium
import numpy as np
import random

from typing import Dict, Union
from gym.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.utils import EzPickle
from PIL import Image
from pprint import pprint
from pettingzoo.utils.conversions import parallel_wrapper_fn, to_parallel
from omarle.movingcompany.env.renderer import GridRenderer
from pettingzoo.utils.wrappers import BaseWrapper

FPS = 20

__all__ = ["env", "parallel_env", "raw_env"]


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "moving_company_v0",
        "render.modes": ["human", "rgb_array", "grid"],
        "is_parallelizable": True,
        "render_fps": FPS,
        "has_manual_policy": True,
    }

    def __init__(self, size: int = 6, seed: int = 42, max_cycles: int = 30, render_mode=None):
        """The init method takes in environment arguments.
        The environment is a sizexsize grid representing two towers
        separated by distance equal to their height.
        3 agents are spawned randomly in the towers or in the space
        seperating the two towers.
        A package is located at the top of the first tower.
        Goals: Agents have to bring it to the top of the second tower
        the fastest way as possible.

        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """

        self.possible_agents = [f"agent_{i}" for i in range(3)]
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.size = size
        self._seed = seed
        self.max_cycles = max_cycles
        self._best_reward = 0
        self.init_grid_environment(seed)
        self.render_mode = render_mode
        self.renderer = GridRenderer(self.size)

        EzPickle.__init__(
            self,
            size=size,
            seed=seed,
            max_cycles=max_cycles,
            render_mode=render_mode
        )

        self.observation_spaces = {agent: Box(
            0, 6, (3**2,), dtype=np.int64, seed=self._seed) for agent in self.possible_agents}

        self.action_spaces = {agent: Discrete(
            7, seed=self._seed) for agent in self.possible_agents}

    def init_grid_environment(self, seed: int):
        self.grid = np.ones((self.size, self.size), dtype=np.int64)
        for i in range(0, self.size):
            for j in range(0, self.size):
                if i == 0 or j == 0 or i == self.size - 1 or j == self.size - 1 \
                        or (i > 1 and i < self.size - 2 and j > 1 and j < self.size - 2) \
                        or (i == 1 and 1 < j and j < self.size - 2):
                    self.grid[i][j] = 0

        self.grid[1][1] = 5  # Setting the package in initial position
        self.grid[1][self.size - 2] = 4
        self.grid[self.size - 2][1] = 4
        self.grid[self.size - 2][self.size - 2] = 4

        agents_counter = len(self.possible_agents)
        agent_condition_positions = [
            (None, 1), (self.size-2, None), (None, self.size-2)]
        self.agents_position = {agent: (None, None)
                                for agent in self.possible_agents}
        while (agents_counter > 0):
            for i in range(1, self.size-1):
                for j in range(1, self.size-1):
                    if self.grid[i][j] == 1:
                        random.seed(seed)
                        if (random.random() > 0.5):
                            if agents_counter > 0:
                                ic, jc = agent_condition_positions[-1]
                                if (ic is not None and i == ic) or (jc is not None and j == jc):
                                    self.agents_position[f"agent_{agents_counter-1}"] = (
                                        i, j)
                                    self.grid[i][j] = 2
                                    agents_counter -= 1
                                    agent_condition_positions.pop()
                                    continue
        self.best_trajectory = []
        for i in range(1, self.size - 1):
            self.best_trajectory += [(i, 1)]
        for j in range(2, self.size - 1):
            self.best_trajectory += [(self.size - 2, j)]
        for i in range(1, self.size - 2):
            self.best_trajectory += [(self.size - 2 - i, self.size - 2)]

    def apply_action(self, agent_name: str, action: int):
        agent_position = self.agents_position[agent_name]

        if action == 0:
            return

        action -= 1

        if 0 <= action and action <= 3:
            direction = self.directions[action]
            targeted_cell_pos = (
                agent_position[0] + direction[0], agent_position[1] + direction[1])
            # move up, move down, move left, move right
            if self.grid[targeted_cell_pos] == 1:
                agent_cell = self.grid[agent_position]
                self.grid[agent_position] = 1
                self.grid[targeted_cell_pos] = agent_cell
                self.agents_position[agent_name] = targeted_cell_pos

        else:
            cross_surrouding_cells = [self.grid[agent_position[0]+direction[0]]
                                      [agent_position[1]+direction[1]] for direction in self.directions]

            # take package
            if action == 4:
                if 5 in cross_surrouding_cells:
                    dir = self.directions[cross_surrouding_cells.index(5)]
                    package_cell_pos = (
                        agent_position[0] + dir[0], agent_position[1] + dir[1])
                    self.grid[package_cell_pos] = 4
                    self.grid[agent_position] = 3

            # drop package
            if action == 5:
                if 4 in cross_surrouding_cells and self.grid[agent_position] == 3:
                    dir = self.directions[cross_surrouding_cells.index(4)]
                    dropzone_cell_pos = (
                        agent_position[0] + dir[0], agent_position[1] + dir[1])
                    self.grid[agent_position] = 2
                    self.grid[dropzone_cell_pos] = 5

    def generate_action_masks(self, agent_name: str):

        action_mask = np.zeros(self.action_space(agent_name).n, dtype=np.int64)

        for action in range(self.action_space(agent_name).n):

            if action == 0:
                action_mask[0] = 1

            elif action in range(1, 4):
                agent_position = self.agents_position[agent_name]

                direction = self.directions[action - 1]
                targeted_cell_pos = (
                    agent_position[0] + direction[0], agent_position[1] + direction[1])

                # move up, move down, move left, move right
                if self.grid[targeted_cell_pos] == 1:
                    action_mask[action] = 1

            else:
                cross_surrouding_cells = [self.grid[agent_position[0]+direction[0]]
                                          [agent_position[1]+direction[1]] for direction in self.directions]

                # take package
                if action == 5:
                    if 5 in cross_surrouding_cells:
                        action_mask[action] = 1

                # drop package
                if action == 6:
                    if 4 in cross_surrouding_cells and self.grid[agent_position] == 3:
                        action_mask[action] = 1

        return action_mask

    def check_terminated(self) -> bool:
        return self.grid[1][-2] == 5

    def compute_reward(self) -> float:

        package_pos = None
        for i in range(0, self.size):
            for j in range(0, self.size):
                if self.grid[i][j] in [3, 5]:
                    package_pos = (i, j)
                    break
            if package_pos is not None:
                break

        for i, pos in enumerate(self.best_trajectory):
            progress_counter = i
            if package_pos == pos:
                break

        progress_difference = progress_counter - self._best_reward
        if (progress_difference > 0):
            self._best_reward = progress_counter
            return progress_counter ** 2

        return progress_difference ** 2

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # An agent sees the neighboring cells (3X3 grid):
        # [ ][ ][ ]
        # [ ][X][ ]
        # [ ][ ][ ]
        # Each cell has 6 possible states: Wall (0), Empty (1), Agent (2), Agent+Package (3), EmptyPackageZone (4), NonEmptyPackageZone (5)
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # An has 6 actions: nothing (0), move up (1), move down (2), move left (3), move right (4), take package (5), drop package (6)
        return Discrete(7, seed=self._seed)

    def compute_pixel_image(self):
        """
        This method should return a pixel representation of the environment.
        """

        pass

    def render(self, mode="human") -> np.ndarray:
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.  
        """

        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == "human":
            # Display pyGame window
            print(self.grid)

        if self.render_mode == "grid":
            return {"grid": self.grid, "agents_position": self.agents_position}

        if self.render_mode == "rgb_array":
            # Generate an image
            return self.renderer.render_grid_frame(self.grid, self.agents_position)

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        agent_pos = self.agents_position[agent]
        observation = [0] * (3**2)
        for i, di in enumerate([-1, 0, 1]):
            for j, dj in enumerate([-1, 0, 1]):
                observation[i * 3 + j] = self.grid[agent_pos[0] +
                                                   di][agent_pos[1]+dj]

        return np.array(observation, dtype=np.int64)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        self.renderer.close()

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """

        self.init_grid_environment(
            seed=seed if seed is not None else self._seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {"action_masks": self.generate_action_masks(
            agent)} for agent in self.agents}
        self.observations = {agent: self.observe(
            agent) for agent in self.agents}
        self.num_cycle = 0
        self._best_reward = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """

        if self.dones[self.agent_selection]:
            self._was_done_step(action)
            return

        agent = self.agent_selection

        self.apply_action(agent, action)

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            common_reward = self.compute_reward()
            # rewards for all agents are placed in the .rewards dictionary
            for ag in self.agents:
                self.rewards[ag] = common_reward

            self.num_cycle += 1
            if self.num_cycle >= self.max_cycles:
                for ag in self.agents:
                    self.dones[ag] = True

            if self.check_terminated():
                for ag in self.agents:
                    self.dones[ag] = True

        elif self._agent_selector.is_first():
            for ag in self.agents:
                self.rewards[ag] = 0

        # observe the current state and generate action masks
        for ag in self.agents:
            self.observations[ag] = self.observe(ag)
            self.infos[agent] = {
                "action_masks": self.generate_action_masks(agent)}

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():

    e = parallel_env(size=10, seed=42, render_mode="rgb_array")

    init_obs = e.reset()

    _perfect_policy = [5, 0, 0, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 6, 0, 0, 0, 5, 0, 0, 4, 0, 0, 4,
                       0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 6, 0, 0, 0, 5, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 6]

    agent_0_policy = [5, 2, 2, 2, 2, 2, 6, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    agent_1_policy = [0, 0, 0, 0, 0, 0, 0, 5,
                      4, 4, 4, 4, 4, 6, 0, 0, 0, 0, 0, 0, 0]
    agent_2_policy = [0, 2, 2, 2, 2, 2, 0, 0,
                      0, 0, 0, 0, 0, 0, 5, 1, 1, 1, 1, 1, 6]

    perfect_policy = {
        "agent_0": agent_0_policy,
        "agent_1": agent_1_policy,
        "agent_2": agent_2_policy
    }

    frame_list = [Image.fromarray(e.render())]

    cumulative_reward = 0
    i = 0
    while e.agents:
        # actions = {agent: e.action_space(agent).sample() for agent in e.agents}
        print("Step: ", i)

        actions = {agent: perfect_policy[agent].pop(0) if len(
            perfect_policy[agent]) > 0 else 0 for agent in e.agents}

        observations, rewards, dones, infos = e.step(
            actions)

        cumulative_reward += rewards["agent_0"]

        img = Image.fromarray(e.render())
        frame_list.append(img)

        i += 1

    print("Cumulative Reward: ", cumulative_reward)

    frame_list[0].save("out.gif", save_all=True,
                       append_images=frame_list[1:], duration=5, loop=0)

    e.close()

    # label_to_obj: Dict = {
    #     "a0": 0,
    #     "a1": 1,
    #     "a2": 2,
    #     "a3": 3,
    #     "a4": 4,
    #     "a5": 5,
    #     "a6": 6,

    #     "o01": np.array([0, 1, 0, 0, 2, 0, 0, 1, 0]),  # 0 -> 1
    #     "o02": np.array([0, 5, 0, 0, 2, 0, 0, 1, 0]),  # 1 -> 5
    #     "o03": np.array([0, 4, 0, 0, 3, 0, 0, 1, 0]),  # 2 -> 2
    #     "o04": np.array([0, 1, 0, 0, 3, 0, 0, 1, 0]),  # 2 -> 2
    #     "o05": np.array([0, 1, 0, 0, 3, 0, 0, 4, 1]),  # 3 -> 6
    #     "o06": np.array([0, 1, 0, 0, 3, 0, 0, 4, 2]),  # 3 -> 6
    #     "o07": np.array([0, 1, 0, 0, 2, 0, 0, 5, 1]),  # -> 0
    #     "o08": np.array([0, 1, 0, 0, 2, 0, 0, 5, 2]),  # -> 0
    #     "o09": np.array([0, 1, 0, 0, 2, 0, 0, 4, 3]),  # -> 0
    #     "o010": np.array([0, 1, 0, 0, 2, 0, 0, 4, 1]),  # -> 0

    #     "o11": np.array([1, 0, 0, 5, 2, 1, 0, 0, 0]),  # 1 -> 5
    #     "o12": np.array([2, 0, 0, 5, 2, 1, 0, 0, 0]),  # 1 -> 5
    #     "o13": np.array([1, 0, 0, 4, 3, 1, 0, 0, 0]),  # 2 -> 4
    #     "o14": np.array([2, 0, 0, 4, 3, 1, 0, 0, 0]),  # 2 -> 4
    #     "o15": np.array([0, 0, 0, 1, 3, 1, 0, 0, 0]),  # 2 -> 4
    #     "o16": np.array([0, 0, 0, 1, 2, 1, 0, 0, 0]),  # 0 -> 3
    #     "o17": np.array([0, 0, 1, 1, 3, 4, 0, 0, 0]),  # 3 -> 6
    #     "o18": np.array([0, 0, 2, 1, 3, 4, 0, 0, 0]),  # 3 -> 6
    #     "o19": np.array([0, 0, 1, 1, 2, 5, 0, 0, 0]),  # -> 0
    #     "o110": np.array([0, 0, 2, 1, 2, 5, 0, 0, 0]),  # -> 0
    #     "o111": np.array([1, 0, 0, 4, 2, 1, 0, 0, 0]),  # -> 0
    #     "o112": np.array([3, 0, 0, 4, 2, 1, 0, 0, 0]),  # -> 0
    #     "o113": np.array([0, 0, 1, 1, 2, 4, 0, 0, 0]),  # -> 0
    #     "o114": np.array([0, 0, 3, 1, 2, 4, 0, 0, 0]),  # -> 0

    #     "o21": np.array([0, 1, 0, 0, 2, 0, 1, 5, 0]),  # 2 -> 5
    #     "o22": np.array([0, 1, 0, 0, 2, 0, 2, 5, 0]),  # 2 -> 5
    #     "o23": np.array([0, 1, 0, 0, 3, 0, 1, 4, 0]),  # 3 -> 1
    #     "o25": np.array([0, 4, 0, 0, 2, 0, 0, 1, 0]),  # 1 -> 2
    #     "o26": np.array([0, 1, 0, 0, 2, 0, 1, 4, 0]),  # -> 0
    #     "o27": np.array([0, 1, 0, 0, 2, 0, 3, 4, 0])  # -> 0
    # }

    # obj_to_label = {str(v): k for k, v in label_to_obj.items()}


if __name__ == '__main__':
    main()

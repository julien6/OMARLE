import gym
import numpy as np
import os
import pygame

from gym.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers
from typing import Dict, Tuple
from copy import deepcopy
from pettingzoo.utils.conversions import from_parallel_wrapper

NUMBER_FONT_SIZE = 28


def raw_env(kwargs): from_parallel_wrapper(parallel_env(**kwargs))


def env(**kwargs):
    env = parallel_env(**kwargs)
    return env


def parallel_env(grid_size=(10, 10), agents_number=3, view_size=3, seed=42, max_cycles=100):
    """
    Factory function to create the Warehouse Management environment.
    Args:
        grid_size (tuple): Size of the grid (rows, cols).
        agents_number (int): Number of agents in the environment.
        view_size (int): Observation range of each agent.
    """
    environment = WarehouseManagementEnv(
        grid_size=grid_size, agents_number=agents_number, view_size=view_size, seed=seed, max_cycles=max_cycles)
    # return wrappers.CaptureStdoutWrapper(environment)
    return environment


class WarehouseManagementEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "warehouse_management_v1"}

    CELL_SIZE = 50  # Taille d'une cellule en pixels
    WINDOW_SIZE = (10 * CELL_SIZE, 10 * CELL_SIZE)  # Taille de la fenêtre

    # Couleurs et images associées aux types de cellules
    COLORS = {
        1: (255, 255, 255),  # EMPTY: white
        0: (128, 128, 128),  # OBSTACLE: grey
        2: (255, 255, 255),  # AGENT_WITHOUT_OBJECT: agent image
        3: (255, 255, 255),  # AGENT_WITH_PRIMARY: agent with primary object image
        # AGENT_WITH_SECONDARY: agent with secondary object image
        4: (255, 255, 255),
        5: (255, 255, 255),  # PRIMARY_OBJECT: primary object image
        6: (255, 255, 255),  # SECONDARY_OBJECT: secondary object image
        7: (173, 216, 230),  # EMPTY_INPUT: light blue
        8: (173, 216, 230),  # INPUT_WITH_OBJECT: input object image
        9: (255, 200, 128),  # EMPTY_INPUT_CRAFT: light orange
        10: (255, 200, 128),  # INPUT_CRAFT_WITH_OBJECT: input craft image
        11: (255, 165, 0),   # EMPTY_OUTPUT_CRAFT: dark orange
        12: (255, 165, 0),   # OUTPUT_CRAFT_WITH_OBJECT: output craft image
        13: (255, 182, 193),  # EMPTY_OUTPUT: light red
        14: (255, 182, 193),  # OUTPUT_WITH_OBJECT: output object image
    }

    IMAGES = {
        2: "asset/agent.png",
        3: "asset/agent_primary_object.png",
        4: "asset/agent_secondary_object.png",
        5: "asset/primary_object.png",
        6: "asset/secondary_object.png",
        7: "asset/input.png",
        8: "asset/primary_object.png",
        9: "asset/input.png",
        10: "asset/primary_object.png",
        11: "asset/output.png",
        12: "asset/secondary_object.png",
        13: "asset/output.png",
        14: "asset/secondary_object.png",
    }

    def __init__(self, grid_size=(10, 10), agents_number=3, view_size=3, seed=42, max_cycles=100):
        """
        Initialize the Warehouse Management environment.
        Args:
            grid_size (tuple): Dimensions of the warehouse grid.
            agents_number (int): Number of agents in the environment.
            view_size (int): Observation range for each agent.
        """
        super().__init__()

        # Environment setup
        self.grid_size = grid_size
        self.agents_number = agents_number
        self.view_size = view_size
        self.max_cycles = max_cycles

        # Agent and action spaces
        self.agents = [f"agent_{i}" for i in range(agents_number)]
        self.possible_agents = self.agents[:]
        # 0: noop, 1: up, 2: down, 3: left, 4: right, 5: pick, 6: drop
        self.action_spaces = {agent: Discrete(
            7, seed=seed) for agent in self.agents}
        self.observation_spaces = {agent: Box(
            0, 14, ((view_size * 2 + 1) ** 2,), np.int64, seed=seed) for agent in self.agents}

        # Cell types (adjusted based on descriptions)
        self.EMPTY = 1
        self.OBSTACLE = 0
        self.AGENT_WITHOUT_OBJECT = 2
        self.AGENT_WITH_PRIMARY = 3
        self.AGENT_WITH_SECONDARY = 4
        self.PRIMARY_OBJECT = 5
        self.SECONDARY_OBJECT = 6
        self.EMPTY_INPUT = 7
        self.INPUT_WITH_OBJECT = 8
        self.EMPTY_INPUT_CRAFT = 9
        self.INPUT_CRAFT_WITH_OBJECT = 10
        self.EMPTY_OUTPUT_CRAFT = 11
        self.OUTPUT_CRAFT_WITH_OBJECT = 12
        self.EMPTY_OUTPUT = 13
        self.OUTPUT_WITH_OBJECT = 14

        self.directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        self.reset()

        # Initialisation de Pygame pour le rendu graphique
        pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("Warehouse Management Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, NUMBER_FONT_SIZE)
        self.loaded_images = {key: pygame.image.load(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), path)) if path else None for key, path in self.IMAGES.items()}
        for key in self.loaded_images:
            if self.loaded_images[key]:
                self.loaded_images[key] = pygame.transform.scale(
                    self.loaded_images[key], (self.CELL_SIZE, self.CELL_SIZE))

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self._initialize_grid()
        self._initialize_agents()
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_cycle = 0
        return self.observe()

    def _initialize_grid(self):
        """
        Create the initial grid layout with zones, obstacles, and objects.
        """
        self.grid = np.full(self.grid_size, self.EMPTY)

        # Define obstacles
        self.grid[4, 4] = self.OBSTACLE
        self.grid[5, 4] = self.OBSTACLE

        # Define input zones
        self.grid[2, 0] = self.EMPTY_INPUT
        self.grid[3, 0] = self.INPUT_WITH_OBJECT
        self.grid[4, 0] = self.INPUT_WITH_OBJECT
        self.grid[5, 0] = self.INPUT_WITH_OBJECT
        self.grid[6, 0] = self.INPUT_WITH_OBJECT
        self.grid[7, 0] = self.EMPTY_INPUT

        # Define craft zones
        self.grid[3, 4] = self.EMPTY_INPUT_CRAFT  # EMPTY_INPUT_CRAFT
        self.grid[6, 4] = self.EMPTY_INPUT_CRAFT
        self.grid[4, 5] = self.EMPTY_OUTPUT_CRAFT
        self.grid[5, 5] = self.EMPTY_OUTPUT_CRAFT

        # Define output zones
        self.grid[4, 9] = self.EMPTY_OUTPUT
        self.grid[5, 9] = self.EMPTY_OUTPUT

    def _initialize_agents(self):
        """
        Place agents in their initial positions.
        """
        self.agent_positions = {
            agent: (i + 4, 2) for i, agent in enumerate(self.agents)
        }
        self.agent_states = {
            agent: self.AGENT_WITHOUT_OBJECT for agent in self.agents}

    def observe(self) -> Dict[str, np.ndarray]:
        """
        Return observations for all agents.
        Each observation is a grid slice centered around the agent.
        """
        observations = {}
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            slice_x = slice(x, x + (2 * self.view_size) + 1)
            slice_y = slice(y, y + (2 * self.view_size) + 1)
            obs = deepcopy(self.grid)
            obs[x, y] = self.agent_states[agent]
            obs = np.pad(obs, pad_width=((self.view_size, self.view_size), (self.view_size, self.view_size)), mode='constant',
                         constant_values=self.OBSTACLE)
            obs = obs[slice_x, slice_y]
            obs_flat = obs.flatten()
            observations[agent] = obs_flat  # padded_obs[..., np.newaxis]
        return observations

    def step(self, actions: Dict[str, int]):
        """
        Apply the actions taken by all agents.
        Args:
            actions (dict): A dictionary mapping agents to their actions.
        """
        previous_state = deepcopy(self.grid)
        for agent, action in actions.items():

            if self.dones[agent]:
                continue

            if action in [1, 2, 3, 4]:  # Movement
                self._move_agent(agent, action)
            elif action == 5:  # Pick up object
                self._pick_object(agent)
            elif action == 6:  # Drop object
                self._drop_object(agent)

        # Compute rewards and check terminations
        self._update_rewards(previous_state)

        self.num_cycle += 1

        for ag in self.agents:
            if self._check_termination(ag):
                self.dones[ag] = True
                self.dones["__all__"] = True

        if self.num_cycle >= self.max_cycles:
            self.dones["__all__"] = True
            for ag in self.agents:
                self.dones[ag] = True

        self.infos = {agent: {} for agent in self.agents}

        return self.observe(), self.rewards, self.dones, self.infos

    def _move_agent(self, agent: str, action: int):
        """
        Moves an agent if the move is valid.
        Args:
            agent (str): Agent identifier.
            action (int): Movement action (1: up, 2: down, 3: left, 4: right).
        """
        x, y = self.agent_positions[agent]
        new_x, new_y = x, y
        if action == 1 and x > 0:  # Up
            new_x -= 1
        elif action == 2 and x < self.grid_size[0] - 1:  # Down
            new_x += 1
        elif action == 3 and y > 0:  # Left
            new_y -= 1
        elif action == 4 and y < self.grid_size[1] - 1:  # Right
            new_y += 1

        if self.grid[new_x, new_y] in [self.EMPTY] and (new_x, new_y) not in self.agent_positions.values():
            self.agent_positions[agent] = (new_x, new_y)

    def _pick_object(self, agent: str):
        """
        Handles picking up objects from valid cells.
        """
        x, y = self.agent_positions[agent]

        for transform_data in [(self.INPUT_CRAFT_WITH_OBJECT, self.AGENT_WITH_PRIMARY, self.EMPTY_INPUT_CRAFT), (self.INPUT_WITH_OBJECT, self.AGENT_WITH_PRIMARY, self.EMPTY_INPUT), (self.OUTPUT_WITH_OBJECT, self.AGENT_WITH_SECONDARY, self.EMPTY_OUTPUT), (self.OUTPUT_CRAFT_WITH_OBJECT, self.AGENT_WITH_SECONDARY, self.EMPTY_OUTPUT_CRAFT), (self.PRIMARY_OBJECT, self.AGENT_WITH_PRIMARY, self.EMPTY),
                               (self.SECONDARY_OBJECT, self.AGENT_WITH_SECONDARY, self.EMPTY)]:
            if self.agent_states[agent] in [self.AGENT_WITHOUT_OBJECT]:
                for direction in self.directions:
                    next_x, next_y = x + direction[0] if 0 <= x + direction[0] and x + direction[0] < self.grid.shape[0] else x, \
                        y + direction[1] if 0 <= y + direction[1] and y + \
                        direction[1] < self.grid.shape[1] else y
                    if self.grid[next_x, next_y] == transform_data[0]:
                        self.agent_states[agent] = transform_data[1]
                        self.grid[next_x, next_y] = transform_data[2]
                        return

    def _drop_object(self, agent: str):
        """
        Handles dropping objects onto valid cells.
        """
        x, y = self.agent_positions[agent]

        if self.agent_states[agent] in [self.AGENT_WITH_PRIMARY]:
            for transform_data in [(self.EMPTY_INPUT_CRAFT, self.AGENT_WITHOUT_OBJECT, self.INPUT_CRAFT_WITH_OBJECT), (self.EMPTY_INPUT, self.AGENT_WITHOUT_OBJECT, self.INPUT_WITH_OBJECT), (self.EMPTY, self.AGENT_WITHOUT_OBJECT, self.PRIMARY_OBJECT)]:

                for direction in self.directions:
                    next_x, next_y = x + direction[0] if 0 <= x + direction[0] and x + direction[0] < self.grid.shape[0] else x, \
                        y + direction[1] if 0 <= y + direction[1] and y + \
                        direction[1] < self.grid.shape[1] else y
                    if self.grid[next_x, next_y] == transform_data[0]:
                        self.agent_states[agent] = transform_data[1]
                        self.grid[next_x, next_y] = transform_data[2]
                        self._update_craft()
                        return

        if self.agent_states[agent] in [self.AGENT_WITH_SECONDARY]:
            for transform_data in [(self.EMPTY_OUTPUT, self.AGENT_WITHOUT_OBJECT, self.OUTPUT_WITH_OBJECT), (self.EMPTY_OUTPUT_CRAFT, self.AGENT_WITHOUT_OBJECT, self.OUTPUT_CRAFT_WITH_OBJECT), (self.EMPTY, self.AGENT_WITHOUT_OBJECT, self.SECONDARY_OBJECT)]:

                for direction in self.directions:
                    next_x, next_y = x + direction[0] if 0 <= x + direction[0] and x + direction[0] < self.grid.shape[0] else x, \
                        y + direction[1] if 0 <= y + direction[1] and y + \
                        direction[1] <= self.grid.shape[1] else y
                    if self.grid[next_x, next_y] == transform_data[0]:
                        self.agent_states[agent] = transform_data[1]
                        self.grid[next_x, next_y] = transform_data[2]
                        return

    def _update_craft(self):
        """
        Update the grid to craft secondary object when all input craft are full
        """
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                cell_value = self.grid[row, col]
                if cell_value == self.EMPTY_INPUT_CRAFT:
                    return

        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                cell_value = self.grid[row, col]
                if cell_value == self.INPUT_CRAFT_WITH_OBJECT:
                    self.grid[row, col] = self.EMPTY_INPUT_CRAFT

        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                cell_value = self.grid[row, col]
                if cell_value == self.EMPTY_OUTPUT_CRAFT:
                    self.grid[row, col] = self.OUTPUT_CRAFT_WITH_OBJECT
                    return

    def _update_rewards(self, previous_state):
        """
        Update rewards based on the state of the environment.
        """
        total_reward = 0
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                cell_value = self.grid[row, col]
                # primary in input
                if cell_value in [self.EMPTY_OUTPUT]:
                    total_reward -= 1

                # primary with agent
                if cell_value == self.AGENT_WITH_PRIMARY and previous_state[row, col] == self.AGENT_WITHOUT_OBJECT:
                    total_reward += 0

                # primary on ground
                if cell_value == self.PRIMARY_OBJECT:
                    total_reward += 0

                # primary in input craft
                if cell_value == self.INPUT_CRAFT_WITH_OBJECT and previous_state[row, col] == self.EMPTY_INPUT_CRAFT:
                    total_reward += 15

                # secondary in output craft
                if cell_value == self.OUTPUT_CRAFT_WITH_OBJECT and previous_state[row, col] == self.EMPTY_OUTPUT_CRAFT:
                    total_reward += 30

                # secondary on ground
                if cell_value == self.SECONDARY_OBJECT:
                    total_reward += 0

                # secondary with agent
                if cell_value == self.AGENT_WITH_SECONDARY and previous_state[row, col] == self.AGENT_WITHOUT_OBJECT:
                    total_reward += 50

                # secondary in output
                if cell_value == self.OUTPUT_WITH_OBJECT and previous_state[row, col] == self.EMPTY_OUTPUT:
                    total_reward += 100

        self.rewards = {agent: reward + total_reward for agent,
                        reward in self.rewards.items()}

    def _check_termination(self, agent: str) -> bool:
        """
        Check whether an agent has completed its task.
        """
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                cell_value = self.grid[row, col]
                if cell_value == self.EMPTY_OUTPUT:
                    return False
        return True

    def render(self, mode="human"):
        """
        Render the current state of the grid.
        """
        pygame.event.get()
        self.screen.fill((0, 0, 0))
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                cell_value = self.grid[row, col]
                cell_rect = pygame.Rect(
                    col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                color = self.COLORS.get(cell_value, (0, 0, 0))
                pygame.draw.rect(self.screen, color, cell_rect)
                if cell_value in self.loaded_images and self.loaded_images[cell_value]:
                    self.screen.blit(
                        self.loaded_images[cell_value], (col * self.CELL_SIZE, row * self.CELL_SIZE))
        for agent, (x, y) in self.agent_positions.items():

            # Render the number as text
            number_text = self.font.render(
                str(agent.split("_")[1]), True, (0, 0, 0))

            agent_image = self.loaded_images.get(
                self.agent_states[agent], self.loaded_images[2]).copy()
            if agent_image:
                agent_image.blit(number_text, (x+18, y+20))
                self.screen.blit(
                    agent_image, (y * self.CELL_SIZE, x * self.CELL_SIZE))

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        """
        Close the environment.
        """
        pygame.quit()


if __name__ == "__main__":

    warehouse_env = env(grid_size=(10, 10), agents_number=3, view_size=3)

    number_to_action = {
        0: "nothing",
        1: "up",
        2: "down",
        3: "left",
        4: "right",
        5: "pick up",
        6: "drop"
    }

    # Mapping des touches pour les actions
    KEY_ACTIONS = {
        pygame.K_UP: 1,     # Move up
        pygame.K_DOWN: 2,   # Move down
        pygame.K_LEFT: 3,   # Move left
        pygame.K_RIGHT: 4,  # Move right
        pygame.K_a: 5,  # Pick up
        pygame.K_q: 6  # Drop
    }

    manual_agent = "agent_0"

    # Réinitialiser l'environnement
    observations = warehouse_env.reset()
    print("=== État initial de la grille ===")
    warehouse_env.render()

    for step in range(200):  # Effectuer 10 étapes
        print(f"\n=== Étape {step + 1} ===")

        # Actions aléatoires pour tous les agents
        actions = {agent: warehouse_env.action_space(
            agent).sample() for agent in warehouse_env.agents}

        # Gestion des événements (clavier)
        running = True
        while (running):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    # Assigner une action à l'agent contrôlé manuellement
                    if event.key in KEY_ACTIONS:
                        actions[manual_agent] = KEY_ACTIONS[event.key]
                        running = False
                        break

        print(f"Actions des agents: ", {
              agent: number_to_action[action] for agent, action in actions.items()})

        # Effectuer une étape
        observations, rewards, terminated, truncated = warehouse_env.step(
            actions)

        # Afficher la grille après l'étape
        warehouse_env.render()

        # Afficher les récompenses
        print(f"Récompenses: {rewards}")

        # Vérifier si tous les agents ont terminé
        if all(terminated.values()):
            print("\nTous les agents ont terminé leurs tâches.")
            break

    warehouse_env.close()

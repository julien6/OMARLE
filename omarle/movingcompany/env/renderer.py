import copy
import pygame
import numpy as np
import os

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)
NUMBER_FONT_SIZE = 28


class GridRenderer:
    """
    Class to render the grid environment in the Moving Company game
    """

    def __init__(self, grid_data_size):

        # Define constants
        self.grid_size = grid_data_size
        self.pixel_cell_size = 40
        self.margin = 1
        self.width = (self.grid_size *
                      (self.pixel_cell_size + self.margin)) - self.margin
        self.height = (self.grid_size *
                       (self.pixel_cell_size + self.margin)) - self.margin

        # Initialize Pygame
        pygame.init()

        # Set up the display
        self.screen = pygame.Surface((self.width, self.height), flags=pygame.SRCALPHA)
        # pygame.display.set_caption("Retro-Game Grid")

        current_dir = os.path.abspath(os.path.dirname(__file__))

        # Load images
        self.agent_box_image = pygame.image.load(
            f'{current_dir}/asset/agent_box.png')#.convert_alpha()
        self.box_image = pygame.image.load(
            f'{current_dir}/asset/box.png')#.convert_alpha()
        self.agent_image = pygame.image.load(
            f'{current_dir}/asset/agent.png')#.convert_alpha()

        self.font = pygame.font.Font(None, NUMBER_FONT_SIZE)

    # Function to draw grid lines
    def draw_grid(self):

        for x in range(0, self.width, self.pixel_cell_size + self.margin):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, self.height))

        for y in range(0, self.height, self.pixel_cell_size + self.margin):
            pygame.draw.line(self.screen, BLACK, (0, y), (self.width, y))

    # Function to draw the grid elements

    def draw_elements(self, grid_data, agents_position):

        for x in range(0, grid_data.shape[0]):
            for y in range(0, grid_data.shape[1]):
                rect = pygame.Rect(x * (self.pixel_cell_size + self.margin), y * (self.pixel_cell_size + self.margin),
                                   self.pixel_cell_size, self.pixel_cell_size)
                if grid_data[x][y] == 0:
                    pygame.draw.rect(self.screen, GRAY, rect)
                elif grid_data[x][y] == 1:
                    pygame.draw.rect(self.screen, WHITE, rect)
                elif grid_data[x][y] == 2:

                    agent, position = [
                        (ag, pos) for ag, pos in agents_position.items() if pos == (y, x)][0]

                    # Render the number as text
                    number_text = self.font.render(
                        str(agent.split("_")[1]), True, BLACK)

                    agent_image = copy.copy(self.agent_image)

                    # Blit the text onto the image
                    agent_image.blit(
                        number_text, (position[0]+7, position[1]+10))

                    pygame.draw.rect(self.screen, WHITE, rect)
                    self.screen.blit(agent_image, rect)
                elif grid_data[x][y] == 3:

                    agent, position = [
                        (ag, pos) for ag, pos in agents_position.items() if pos == (y, x)][0]

                    # Render the number as text
                    number_text = self.font.render(
                        str(agent.split("_")[1]), True, BLACK)

                    agent_box_image = copy.copy(self.agent_box_image)

                    # Blit the text onto the image
                    agent_box_image.blit(
                        number_text, (position[0]+7, position[1]+10))

                    pygame.draw.rect(self.screen, WHITE, rect)
                    self.screen.blit(agent_box_image, rect)
                elif grid_data[x][y] == 4:
                    pygame.draw.rect(self.screen, ORANGE, rect)
                elif grid_data[x][y] == 5:
                    # pygame.draw.rect(self.screen, RED, rect)
                    self.screen.blit(self.box_image, rect)

    def render_grid_frame(self, grid_data, agent_positions) -> np.ndarray:

        # # Main game loop
        # running = True
        # while running:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False

        grid_data = np.rot90(grid_data)
        grid_data = np.flip(grid_data, 0)

        # Clear the self.screen
        self.screen.fill(BLACK)

        # Draw grid lines
        self.draw_grid()

        # Draw grid elements
        self.draw_elements(grid_data, agent_positions)

        # # Update the display
        # pygame.display.flip()

        # Récupérer l'image sous forme de tableau numpy
        rgb_array = pygame.surfarray.array3d(self.screen)

        rgb_array = np.rot90(rgb_array)
        rgb_array = np.flip(rgb_array, 0)

        return rgb_array

    def close(self):
        pygame.quit()

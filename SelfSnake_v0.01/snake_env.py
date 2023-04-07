import pygame
import random
import numpy as np


class SnakeEnv:
    def __init__(self, width, height, size):
        self.width = width
        self.height = height
        self.size = size
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [[self.width // 2, self.height // 2]]
        self.direction = None
        while self.direction is None:
            direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
            if (
                (direction == "UP" and self.snake[0][1] != self.height - self.size)
                or (direction == "DOWN" and self.snake[0][1] != 0)
                or (direction == "LEFT" and self.snake[0][0] != self.width - self.size)
                or (direction == "RIGHT" and self.snake[0][0] != 0)
            ):
                self.direction = direction

        self.food = [random.randrange(0, self.width // self.size) * self.size, random.randrange(0, self.height // self.size) * self.size]
        self.score = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        if action == 0 and self.direction != "DOWN":
            self.direction = "UP"
        elif action == 1 and self.direction != "UP":
            self.direction = "DOWN"
        elif action == 2 and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif action == 3 and self.direction != "LEFT":
            self.direction = "RIGHT"

        if self.direction == "UP":
            self.snake[0][1] -= self.size
        elif self.direction == "DOWN":
            self.snake[0][1] += self.size
        elif self.direction == "LEFT":
            self.snake[0][0] -= self.size
        elif self.direction == "RIGHT":
            self.snake[0][0] += self.size

        # Add a check to ensure the snake's position is within the valid range
        if not (0 <= self.snake[0][0] < self.width and 0 <= self.snake[0][1] < self.height) or any([self.snake[0] == s for s in self.snake[1:]]):
            self.done = True
            return self.get_state(), -1, self.done

        if self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]:
            self.food = [random.randrange(0, int(self.width / self.size)) * self.size, random.randrange(0, int(self.height / self.size)) * self.size]
            self.score += 1
            self.snake.append([self.snake[-1][0], self.snake[-1][1]])
            reward = 1
        else:
            reward = 0

        for i in range(len(self.snake) - 1, 0, -1):
            self.snake[i][:] = self.snake[i - 1][:]

        return self.get_state(), reward, self.done

    def get_state(self):
        state = np.zeros((self.height, self.width), dtype=int)
        for x, y in self.snake:
            state[y][x] = 1
        state[self.food[1]][self.food[0]] = 2
        return state

    def render(self, delay=100):
        self.screen.fill((0, 0, 0))

        # Draw food
        pygame.draw.rect(self.screen, (0, 255, 0), [self.food[0], self.food[1], self.size, self.size])

        # Draw snake
        for x, y in self.snake:
            pygame.draw.rect(self.screen, (255, 255, 255), [x, y, self.size, self.size])

        pygame.display.flip()
        self.clock.tick(1000 // delay)

        # Add an event handling loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit(0)

    def close(self):
        pygame.quit()

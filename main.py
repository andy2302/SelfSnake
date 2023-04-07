import pygame
import random

# Initialize Pygame
pygame.init()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Set the screen
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
SIZE = 20

# Set the caption
pygame.display.set_caption("Snake Game")

# Define the clock
clock = pygame.time.Clock()

# Define the font
font = pygame.font.SysFont(None, 30)

# Define the snake
# snake = []
snake = [[WIDTH / 2, HEIGHT / 2]]
# snake.append([WIDTH / 2, HEIGHT / 2])
# snake.append([WIDTH / 2, HEIGHT / 2 + SIZE])
# snake.append([WIDTH / 2, HEIGHT / 2 + SIZE * 2])

# Define the direction
direction = None

# Wait for a key press to start the game and set initial direction
while direction is None:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                direction = "UP"
            elif event.key == pygame.K_DOWN:
                direction = "DOWN"
            elif event.key == pygame.K_LEFT:
                direction = "LEFT"
            elif event.key == pygame.K_RIGHT:
                direction = "RIGHT"

# Define the food
food = [random.randrange(0, WIDTH // SIZE) * SIZE, random.randrange(0, HEIGHT // SIZE) * SIZE]
# Define the poison
poison = [random.randrange(0, WIDTH // SIZE) * SIZE, random.randrange(0, HEIGHT // SIZE) * SIZE]

# Define the score
score = 0

# Main game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != "DOWN":
                direction = "UP"
            elif event.key == pygame.K_DOWN and direction != "UP":
                direction = "DOWN"
            elif event.key == pygame.K_LEFT and direction != "RIGHT":
                direction = "LEFT"
            elif event.key == pygame.K_RIGHT and direction != "LEFT":
                direction = "RIGHT"

    # Move the snake
    if direction == "UP":
        snake[0][1] -= SIZE
    elif direction == "DOWN":
        snake[0][1] += SIZE
    elif direction == "LEFT":
        snake[0][0] -= SIZE
    elif direction == "RIGHT":
        snake[0][0] += SIZE

    # Check if the snake has hit the wall
    if snake[0][0] < 0 or snake[0][0] >= WIDTH or snake[0][1] < 0 or snake[0][1] >= HEIGHT:
        pygame.quit()
        quit()

    # Check if the snake has hit itself
    for i in range(1, len(snake)):
        if snake[0][0] == snake[i][0] and snake[0][1] == snake[i][1]:
            pygame.quit()
            quit()

    # Check if the snake has eaten the food or poison
    if snake[0][0] == food[0] and snake[0][1] == food[1]:
        food = [random.randrange(0, int(WIDTH / SIZE)) * SIZE, random.randrange(0, int(HEIGHT / SIZE)) * SIZE]
        poison = [random.randrange(0, int(WIDTH / SIZE)) * SIZE, random.randrange(0, int(HEIGHT / SIZE)) * SIZE]
        score += 1
        snake.append([snake[-1][0], snake[-1][1]])
        print(score)
    elif snake[0][0] == poison[0] and snake[0][1] == poison[1]:
        food = [random.randrange(0, int(WIDTH / SIZE)) * SIZE, random.randrange(0, int(HEIGHT / SIZE)) * SIZE]
        poison = [random.randrange(0, int(WIDTH / SIZE)) * SIZE, random.randrange(0, int(HEIGHT / SIZE)) * SIZE]
        score -= 1
        snake.pop()
        print(score)

    # Move the tail
    for i in range(len(snake) - 1, 0, -1):
        snake[i][0] = snake[i - 1][0]
        snake[i][1] = snake[i - 1][1]

    # Draw the background
    screen.fill(BLACK)

    # Draw the food and poison
    pygame.draw.rect(screen, GREEN, [food[0], food[1], SIZE, SIZE])
    if score > 10:
        pygame.draw.rect(screen, RED, [poison[0], poison[1], SIZE, SIZE])

    # Draw the snake
    for i in range(len(snake)):
        pygame.draw.rect(screen, WHITE, [snake[i][0], snake[i][1], SIZE, SIZE])

    # Draw the scoreboard
    score_text = font.render("Score: {}".format(score), True, (255, 255, 255))
    length_text = font.render("Length: {}".format(len(snake)), True, (255, 255, 255))
    position_text = font.render("Position: ({}, {})".format(snake[0][0], snake[0][1]), True, (255, 255, 255))
    screen.blit(score_text, (10, 10))
    screen.blit(length_text, (10, 40))
    screen.blit(position_text, (10, 70))

    # Update the screen
    pygame.display.flip()

    # Add a delay
    pygame.time.delay(100)

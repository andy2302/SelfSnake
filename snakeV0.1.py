import pygame
import random

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

WIDTH, HEIGHT = 1080, 720
SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

snake = [[WIDTH / 2, HEIGHT / 2]]
direction = None

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

food = [random.randrange(0, WIDTH // SIZE) * SIZE, random.randrange(0, HEIGHT // SIZE) * SIZE]
poison = [random.randrange(0, WIDTH // SIZE) * SIZE, random.randrange(0, HEIGHT // SIZE) * SIZE]
score = 0

def read_high_score():
    try:
        with open("high_score.txt", "r") as f:
            return int(f.read())
    except FileNotFoundError:
        return 0

def save_high_score(score):
    with open("high_score.txt", "w") as f:
        f.write(str(score))

high_score = read_high_score()

while True:
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

    if direction == "UP":
        snake[0][1] -= SIZE
    elif direction == "DOWN":
        snake[0][1] += SIZE
    elif direction == "LEFT":
        snake[0][0] -= SIZE
    elif direction == "RIGHT":
        snake[0][0] += SIZE

    if not (0 <= snake[0][0] < WIDTH and 0 <= snake[0][1] < HEIGHT):
        if score > high_score:
            save_high_score(score)
        pygame.quit()
        quit()

    for i in range(1, len(snake)):
        if snake[0][0] == snake[i][0] and snake[0][1] == snake[i][1]:
            if score > high_score:
                save_high_score(score)
            pygame.quit()
            quit()

    if snake[0][0] == food[0] and snake[0][1] == food[1]:
        food = [random.randrange(0, int(WIDTH / SIZE)) * SIZE, random.randrange(0, int(HEIGHT / SIZE)) * SIZE]
        poison = [random.randrange(0, int(WIDTH / SIZE)) * SIZE, random.randrange(0, int(HEIGHT / SIZE)) * SIZE]
        score += 1
        snake.append([snake[-1][0], snake[-1][1]])
        print(score)
    elif score > 10 and snake[0][0] == poison[0] and snake[0][1] == poison[1]:
        food = [random.randrange(0, int(WIDTH / SIZE)) * SIZE, random.randrange(0, int(HEIGHT / SIZE)) * SIZE]
        poison = [random.randrange(0, int(WIDTH / SIZE)) * SIZE, random.randrange(0, int(HEIGHT / SIZE)) * SIZE]
        score -= 1
        snake.pop()
        print(score)

    for i in range(len(snake) - 1, 0, -1):
        snake[i][:] = snake[i - 1][:]

    screen.fill(BLACK)

    pygame.draw.rect(screen, GREEN, [food[0], food[1], SIZE, SIZE])
    if score > 10:
        pygame.draw.rect(screen, RED, [poison[0], poison[1], SIZE, SIZE])

    for x, y in snake:
        pygame.draw.rect(screen, WHITE, [x, y, SIZE, SIZE])

    score_text = font.render("Score: {}".format(score), True, (255, 255, 255))
    length_text = font.render("Length: {}".format(len(snake)), True, (255, 255, 255))
    position_text = font.render("Position: ({}, {})".format(snake[0][0], snake[0][1]), True, (255, 255, 255))
    high_score_text = font.render("High Score: {}".format(high_score), True, (255, 255, 255))

    screen.blit(score_text, (10, 10))
    screen.blit(length_text, (10, 40))
    screen.blit(position_text, (10, 70))
    screen.blit(high_score_text, (10, 100))

    pygame.display.flip()

    pygame.time.delay(100)


import gymnasium as gym

import pygame


# Pallette:
COLOR_1 = (29, 91, 121)
COLOR_2 = (70, 139, 151)
COLOR_3 = (239, 98, 98)
COLOR_4 = (243, 170, 96)


def make_env(env_id, seed, idx, capture_video, run_name, **kwargs):
    def thunk():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


def strtobool(val):
    val = val.lower()
    if val in {'y', 'yes', 't', 'true', 'on', '1'}:
        return True
    elif val in {'n', 'no', 'f', 'false', 'off', '0'}:
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


def draw_agent(screen, x, y, block_size):

    # Calculate the dimensions of the car based on the block_size
    car_width = int(0.8 * block_size)
    car_height = int(0.5 * block_size)
    wheel_radius = int(0.14 * block_size)

    x = x + (block_size - car_width)//2
    y = y + (block_size - car_height)//2

    # Draw the car body
    car_rect = pygame.Rect(x, y, car_width, car_height)
    pygame.draw.rect(screen, COLOR_1, car_rect)

    # Draw the wheels
    left_wheel_center = (x + int(0.25 * car_width), y + car_height)
    right_wheel_center = (x + int(0.75 * car_width), y + car_height)
    pygame.draw.circle(screen, COLOR_3, left_wheel_center, wheel_radius)
    pygame.draw.circle(screen, COLOR_3, right_wheel_center, wheel_radius)


def draw_object(screen, x, y, block_size, is_taken=False):

    # Calculate the dimensions of the object based on the block_size
    object_width = int(0.6 * block_size)
    object_height = int(0.6 * block_size)

    x = x + (block_size - object_width)//2
    y = y + (block_size - object_height)//4

    # Draw a box-like object, add a border if it is taken

    object_rect = pygame.Rect(x, y, object_width, object_height)
    pygame.draw.rect(screen, COLOR_4, object_rect)

    if is_taken:
        border_rect = pygame.Rect(x, y, object_width, object_height)
        pygame.draw.rect(screen, COLOR_2, object_rect, 4)


# draw a cross at the destination
def draw_destination(screen, x, y, block_size):

    # Calculate the dimensions of the destination based on the block_size
    dest_width = int(0.5 * block_size)
    dest_height = int(0.5 * block_size)

    x = x + (block_size - dest_width)//2
    y = y + (block_size - dest_height)//2

    # Draw the cross
    pygame.draw.line(screen, COLOR_3, (x, y),
                     (x + dest_width, y + dest_height), 4)
    pygame.draw.line(screen, COLOR_3, (x + dest_width, y),
                     (x, y + dest_height), 4)

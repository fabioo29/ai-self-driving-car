import os
import math
import pygame


def scale(image, width: int = 30, height: int = 15):
    """Scale the image to the specified width and height."""
    image = pygame.transform.scale(image, (width, height))
    return image


def rotate(image, angle):
    """Rotate the image to the specified angle."""
    image = pygame.transform.rotate(image, angle)
    return image


def blit_rotate(win, image, top_left, angle, crashed: bool = False):
    rotated_image = pygame.transform.rotate(image, angle)
    if crashed:
        rotated_image.set_alpha(50)
    new_rect = rotated_image.get_rect(
        center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)


def get_distance_from_points(x1, y1, x2, y2):
    """get the px distance from the car and the closest track border in the sensor direction"""
    return math.sqrt((int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2)


def get_assets(folder, files):
    """get all assets with name starting with files... from the specified folder"""
    assets_path = os.path.join(folder)

    assets = []
    for file in sorted(os.listdir(assets_path)):
        if files in file:
            elem = pygame.image.load(os.path.join(assets_path, file))
            if files == 'car':
                elem = elem.convert_alpha()
            assets.append(elem)
    return assets


def get_mask(images):
    """get a mask from the images"""
    return [pygame.mask.from_surface(image) for image in images]

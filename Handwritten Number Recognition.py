import pygame
import time
from numpy import array
from PIL import Image, ImageOps
import TensorCore as tc


white = (255, 255, 255)
black = (0, 0, 0)
mouseClickPos = (0, 0)
pygame.init()
tc.singleInstance.LoadTrainedModel()
gameDisplay = pygame.display.set_mode((630, 630), 0, 8)  # Set 630x630 8 bit window
exitFlag = False
gameDisplay.fill(white)


def predict():
    mat = pygame.surfarray.array2d(gameDisplay)  # Get py game window image as 630x630 matrix
    mat = (mat - 255)  # Invert values
    img = Image.fromarray(mat)  # Convert mat to image
    newSize = (28, 28)
    img = img.resize(newSize)  # Resize image to 28x28
    img = ImageOps.mirror(img).rotate(90)  # Flip and rotate image to correct orientation
    mat2 = array(img)  # convert back to mat
    mat2 = mat2.reshape((1, 28, 28, 1))  # reshaping image to give it to classifier
    tc.singleInstance.PredictNumber(mat2)


while not exitFlag:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Exit application on closing window
            exitFlag = True

        if pygame.mouse.get_pressed()[0] == 1:  # Start drawing when mouse clicked
            mouseClickPos = pygame.mouse.get_pos()
            pygame.draw.rect(gameDisplay, black, [mouseClickPos[0], mouseClickPos[1], 30, 30])

        if event.type == pygame.MOUSEBUTTONUP:  # Make prediction when mouse released
            predict()
            gameDisplay.fill(white)
    pygame.display.update()

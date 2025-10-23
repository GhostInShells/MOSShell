import os
import pygame
import live2d.v3 as live2d
from os.path import join, dirname
from live2d.v2 import StandardParams
from ghoshell_moss.channels import PyChannel
import random

model: live2d.LAppModel | None = None


def init_model():
    global model
    model_path = join(dirname(__file__), "models/hiyori/runtime/hiyori_pro_t11.model3.json")
    live2d.init()
    live2d.glInit()
    model = live2d.LAppModel()
    model.LoadModelJson(model_path)


def main():
    pygame.init()

    display = (300, 400)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("pygame window")

    init_model()
    model.Resize(*display)
    model.SetAutoBlinkEnable(True)
    model.SetAutoBreathEnable(True)
    print("+++++++++++", model.GetMotionGroups(), model.GetExpressionIds())

    model.SetParameterValue(StandardParams.ParamEyeLOpen, 0)
    model.SetParameterValue(StandardParams.ParamAngleX, 100)
    model.SetParameterValue(StandardParams.ParamAngleY, 100)
    model.SetParameterValue(StandardParams.ParamShoulderX, 100)
    model.SetParameterValue(StandardParams.ParamBustX, 100)

    running = True
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEMOTION:
                # model.Drag(*pygame.mouse.get_pos())
                pass
            elif event.type == pygame.MOUSEBUTTONUP:
                # model.SetRandomExpression()
                # model.StartRandomMotion()
                pass

        if not running:
            break

        model.SetParameterValue('ParamArmRA', random.randint(-100, 100))
        model.SetParameterValue('ParamArmLA', random.randint(-100, 100))
        model.SetParameterValue('ParamShoulderY', random.randint(-10, 10))
        model.SetParameterValue('ParamHandL', random.randint(-10, 10))
        model.SetParameterValue('ParamHandR', random.randint(-10, 10))
        model.SetParameterValue('ParamAngleX', random.randint(-100, 100))
        model.SetParameterValue('ParamAngleY', random.randint(-100, 100))
        model.SetParameterValue('ParamBaseX', random.randint(-10, 10))
        model.SetParameterValue('ParamBaseY', random.randint(-10, 10))
        model.SetParameterValue('ParamBodyAngleX', random.randint(-10, 10))
        model.SetParameterValue('ParamBodyAngleY', random.randint(-10, 10))
        model.SetParameterValue('ParamBodyAngleZ', random.randint(-10, 10))
        live2d.clearBuffer()
        model.Update()

        model.Draw()

        pygame.time.wait(50)
        pygame.display.flip()

    live2d.dispose()

    pygame.quit()


if __name__ == "__main__":
    main()

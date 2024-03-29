import gym
import matplotlib
import matplotlib.pyplot as plt 
import os
import numpy as np
import cv2
from math import ceil
import scipy
import tensorflow as tf
import argparse


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")

    parser.add_argument(
        "--model",
        type=str,
        default='Resnet',
        help="Select model between Resnet50 and VGG16",
    )

    args = parser.parse_args()
    return args

def preprocess(image):
    img_width, img_height = 224, 224
    image = cv2.resize(image, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)
    np_image = np.array(image).astype('float32')/255
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

if __name__ == "__main__":    
    args = parse_config()
    # load pre-trained model
    if (args.model == 'Resnet'):
        model_path = './../Trained_Model/model_resnet.h5'
        print ("Loading Resnet50 model")
    if (args.model == 'Vgg'):
        model_path = './../Trained_Model/model_vgg.h5'
        print ("Loading VGG16 model")

    loaded_model = tf.keras.models.load_model(model_path)
    plt.ion()

    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    observation, info = env.reset(seed=42)
    cam = cv2.VideoCapture(0)
    while True:
        check, frame = cam.read()
        preic_image = preprocess(frame)
        predic = loaded_model.predict(preic_image)
        action = predic.argmax()
        observation, reward, terminated, truncated, info = env.step(action)
        text = f"Action: {action}, Obesrvation: {observation}"
        if terminated:
            env.reset()
            print(f"Action: {action}, Obesrvation: {observation}, Reward {reward}, Info: {info}, Terminated: {terminated}\n")
            env.close()
            break
        coordinates = (210,470)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0,255,255)
        thickness = 2
        image = cv2.putText(frame, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        env_screen = env.render()
        plt.imshow(env_screen)
        cv2.imshow('video', image)
        key = cv2.waitKey(1)
        if key == 27:
            env.reset()
            env.close()
            break
        plt.show()
        plt.clf()

    cam.release()
    cv2.destroyAllWindows()
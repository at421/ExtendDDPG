import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw 

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Average reward across 10 training cycles')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.savefig(figure_file)


def _label_with_episode_number(frame, model_name):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'{model_name}', fill=text_color)

    return im


def save_model_gif(env, agent, model_name):
    frames = []
    observation, info = env.reset()  
    for t in range(500):
        action = agent.choose_action(observation)

        frame = env.render()
        frames.append(_label_with_episode_number(frame, model_name))

        observation_, reward, done, trunc, info = env.step(action)
        if done:
            break

    env.close()

    imageio.mimwrite(os.path.join('./videos/', f'{model_name}.gif'), frames, fps=60)
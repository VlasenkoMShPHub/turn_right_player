import pyautogui
import time
from PIL import ImageGrab
import cv2
import numpy as np
from direct_keys import PressKey, ReleaseKey, W, A, S, D, left_mouse, right_mouse
from math import sqrt, pow


class Box:
    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = low
        self.high = high


class Env:
    observation_space = Box([3], [0, 0, 0], [600, 400, 1])  # coords, mouse_up
    action_space = Box([1], [0], [1])
    mouse_up = True
    ep_start = 0
    time_checked = 0

    def turn(self, action):
        if self.mouse_up and action >= 0.5:
            # PressKey(0x100)
            PressKey(W)
            self.mouse_up = False
            # print('turning')
        if not self.mouse_up and action < 0.5:
            # ReleaseKey(left_mouse)
            ReleaseKey(W)
            self.mouse_up = True
            # print('stop turning')

    def done(self, image):
        if image[66, 274] == 0:
            return True
        return False

    def process_img(self, image):
        flag = False
        rows, cols = np.where(image == 0)
        # print(f'rows = {rows}, {len(rows)}')
        # print(f'cols = {cols}, {len(cols)}')
        if rows.size == 0 or cols.size == 0:
            flag = True
        row = np.mean(rows)
        col = np.mean(cols)
        return round(row), round(col), flag

    def screen_record(self):
        last_time = time.time()
        while True:
            screen = np.array(ImageGrab.grab(bbox=(400, 50, 800, 650)))
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            if self.done(screen):
                print('done')
                break
            car_row, car_col, is_nan = self.process_img(screen)
            print(f'car on {car_row} {car_col}')
            # screen2 = cv2.Canny(screen, threshold1=200, threshold2=300)
            # process_img(screen)

            print('loop took {} seconds'.format(time.time()-last_time))
            cv2.imshow('window', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            last_time = time.time()

    def get_reward(self, row, col):
        reward = 0
        reward += pow(time.time() - self.ep_start, 1/10)
        if row < 200:
            if row < 85:
                reward += row - 85
            reward += 0.5
        if row > 410:
            if row > 525:
                reward += 515 - row
            reward += 0.5
        if 230 < row < 390 and col < 120:
            reward -= 0.1 * abs(col - 50)
        if 230 < row < 390 and col > 200:
            reward -= 0.1 * abs(col - 165)
        if not self.mouse_up:
            if row < 200 or row > 410:
                reward += 0.5
            if 230 < row < 390:
                reward -= 0.3
        return reward

    def reset(self):
        print('reset')
        self.turn(0)
        screen = np.array(ImageGrab.grab(bbox=(400, 50, 800, 650)))
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        cnt = 0
        while not self.done(screen) and cnt < 10:
            time.sleep(0.1)
            screen = np.array(ImageGrab.grab(bbox=(400, 50, 800, 650)))
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            cnt += 1
        pyautogui.click(800, 400)
        time.sleep(0.1)
        row, col, is_nan = self.process_img(screen)
        state = [row, col, float(self.mouse_up)]
        self.ep_start = time.time()
        self.time_checked = self.ep_start
        print('exit_reset')
        return state

    def step(self, action, act=True):
        is_nan = False
        if act:
            self.turn(float(action))
        screen = np.array(ImageGrab.grab(bbox=(400, 50, 800, 650)))
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        done = self.done(screen)
        if not done:
            car_row, car_col, is_nan = self.process_img(screen)
            # print(f'car on {car_row} {car_col}')
            state = [car_row, car_col, float(self.mouse_up)]
            reward = self.get_reward(car_row, car_col)
        else:
            state = [430.0, 62.0, True]
            reward = -100
        info = 0
        if is_nan:
            if act:
                time.sleep(0.1)
                state, reward, done, info = self.step(action, False)
            else:
                time.sleep(0.2)
                done = True
                state = [164.0, 237.0, False]
                reward = -100
        return state, reward, done, info

'''
env = Env()
time.sleep(3)
pyautogui.click(800, 400)
time.sleep(0.5)
env.turn(0.5)
time.sleep(1.5)
env.turn(0.01)
'''

import pyautogui
import time
from PIL import ImageGrab
import cv2
import numpy as np
from direct_keys import PressKey, ReleaseKey, W, A, S, D, left_mouse, right_mouse
from math import sqrt, pow, atan
from mss import mss


class Box:
    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = low
        self.high = high


class Env:
    observation_space = Box([5], [0, 0, 0, 0, 0], [600, 400, 600, 400, 1])  # coords, mouse_up
    action_space = Box([1], [0], [1])
    mouse_up = True
    ep_start = 0
    time_checked = 0
    have_turned = False
    prev_state = []

    def turn(self, action):
        if self.mouse_up and action >= 0.5:
            # PressKey(0x100)
            PressKey(W)
            self.mouse_up = False
            self.have_turned = True
        if not self.mouse_up and action < 0.5:
            # ReleaseKey(left_mouse)
            ReleaseKey(W)
            self.mouse_up = True

    def done(self, image):
        if image[66, 274] == 0:
            return True
        return False

    def get_screen(self):
        with mss() as sct:
            monitor = {"left": 400, "top": 50, "width": 400, "height": 600}
            screen = sct.grab(monitor)
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2GRAY)
        return screen

    def process_img(self, image):
        flag = False
        rows, cols = np.where(image == 0)
        # print(f'rows = {rows}, {len(rows)}')
        # print(f'cols = {cols}, {len(cols)}')
        if rows.size == 0 or cols.size == 0:
            flag = True
        if flag:
            return 374, 54, flag
        row = np.mean(rows)
        col = np.mean(cols)
        return int(row), int(col), flag

    def get_direction(self, x, y):
        if x == 0:
            return 0
        t = round(atan(y / x) + 2, 1)
        if x < 0:
            return int(round(t + 4, 1) * 10)
        return int(t * 10)

    def get_reward(self, row, col, dir, done=False):
        reward = 3
        ep_time = time.time() - self.ep_start
        # reward += pow(ep_time, 1/10)
        # reward += 2
        '''
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
        '''
        if not self.mouse_up:
            if row < 210 or row > 430:
                reward += 10
                if row < 140 or row > 500:
                    reward += 20
        else:
            if row < 140 or row > 500:
                reward -= 15
            if 230 < row < 400 and 17 <= dir <= 23:
                reward += 10

        # add reward for going straight

        #if self.mouse_up:
            #if 250 < row < 360:
                #reward += 1

        if done:
            reward = -100
            if not self.have_turned:
                reward -= 100
            if ep_time < 1:
                reward -= 50
        return reward

    def reset(self):
        print('reset')
        self.turn(-1)
        screen = self.get_screen()
        cnt = 0
        time.sleep(2)
        # pyautogui.click(800, 400)
        PressKey(W)
        time.sleep(0.2)
        ReleaseKey(W)
        time.sleep(1)
        row, col, is_nan = self.process_img(screen)
        state = [row, col, 0, int(self.mouse_up)]
        self.ep_start = time.time()
        self.time_checked = self.ep_start
        self.have_turned = False
        self.prev_state = [row, col]
        print('exit_reset')
        return state

    def step(self, action, act=True):
        is_nan = False
        if act:
            self.turn(float(action))
        screen = self.get_screen()
        done = self.done(screen)
        if not done:
            car_row, car_col, is_nan = self.process_img(screen)
            state = [car_row, car_col, self.get_direction(
                car_row - self.prev_state[0], car_col - self.prev_state[1]), int(self.mouse_up)]
            reward = self.get_reward(car_row, car_col, state[2])
            self.prev_state = [car_row, car_col]
        else:
            state = [430, 62, 0, 0]
            reward = self.get_reward(0, 0, 20, done=True)
        info = dict()
        info['time'] = time.time() - self.ep_start
        if is_nan:
            time.sleep(0.3)
            done = True
            state = [430, 62, 0, 0]
            reward = self.get_reward(0, 0, 20, done=True)
        return state, reward, done, info

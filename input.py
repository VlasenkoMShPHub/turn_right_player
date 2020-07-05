import time
import cv2
import numpy as np
from direct_keys import PressKey, ReleaseKey, Q, W
from math import atan
from mss import mss


class Box:
    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = low
        self.high = high


class Env:
    def __init__(self):
        self.observation_space = Box([5], [0, 0, 0, 0, 0], [600, 400, 100, 1])  # coords, direction, mouse_up
        self.action_space = Box([1], [0], [1])
        self.mouse_up = True
        self.ep_start = 0
        self.time_checked = 0
        self.have_turned = False
        self.prev_state = []

    @staticmethod
    def done(image):
        if image[330, 225] != 255:
            return True
        return False

    @staticmethod
    def get_screen():
        with mss() as sct:
            monitor = {"left": 0, "top": 30, "width": 400, "height": 600}
            screen = sct.grab(monitor)
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2GRAY)
        return screen

    @staticmethod
    def process_img(image):
        flag = False
        rows, cols = np.where(image == 0)
        if rows.size == 0 or cols.size == 0:
            flag = True
        if flag:
            return 265, 97, flag
        row = np.mean(rows)
        col = np.mean(cols)
        return int(row), int(col), flag

    @staticmethod
    def get_direction(x, y):
        if x == 0:
            return 0
        t = round(atan(y / x) + 2, 1)
        if x < 0:
            return int(round(t + 4, 1) * 10)
        return int(t * 10)

    @staticmethod
    def gate(var1, var2, border):
        if var1 < border <= var2:
            print(f'gate passed {border}')
            return 101
        if var1 > border >= var2:
            print(f'gate passed {border}')
            return 101
        return 0

    @classmethod
    def check_car_pos(cls):
        screen = cls.get_screen()
        row, col, is_nan = cls.process_img(screen)
        return row, col

    def get_reward(self, row, col, direction, done=False):
        if done:
            reward = -100
            if not self.have_turned:
                reward -= 100
            return reward
        reward = -1
        reward += self.gate(self.prev_state[0], row, 145)
        reward += self.gate(self.prev_state[0], row, 330)
        reward += self.gate(self.prev_state[0], row, 515)
        reward += self.gate(self.prev_state[1], col, 225)
        return reward

    def turn(self, action):
        if self.mouse_up and action >= 0.5:
            PressKey(W)
            self.mouse_up = False
            self.have_turned = True
        if not self.mouse_up and action < 0.5:
            ReleaseKey(W)
            self.mouse_up = True

    def reset(self):
        print('reset')
        self.turn(-1)
        time.sleep(1.3)
        PressKey(Q)
        screen = self.get_screen()
        time.sleep(0.05)
        row, col, is_nan = self.process_img(screen)
        state = [row, col, 0, int(self.mouse_up)]
        self.ep_start = time.time()
        self.time_checked = self.ep_start
        self.have_turned = False
        self.prev_state = [row, col]
        ReleaseKey(Q)
        print('exit_reset')
        return state

    def step(self, action, act=True):
        if act:
            self.turn(float(action))
        screen = self.get_screen()
        done = self.done(screen)
        if not done:
            car_row, car_col, is_nan = self.process_img(screen)
            if is_nan:
                state, reward, done, info = self.step(action, act)
            else:
                state = [car_row, car_col, self.get_direction(
                    car_row - self.prev_state[0], car_col - self.prev_state[1]), int(self.mouse_up)]
                reward = self.get_reward(car_row, car_col, state[2])
                self.prev_state = [car_row, car_col]
        else:
            state = [269, 116, 0, 0]
            reward = self.get_reward(0, 0, 0, done=True)
        info = dict()
        info['time'] = time.time() - self.ep_start
        return state, reward, done, info

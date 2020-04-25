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
        if flag:
            return 374, 54, flag
        row = np.mean(rows)
        col = np.mean(cols)
        return int(row), int(col), flag

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

    def get_reward(self, row, col, done=False):
        reward = 0
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
            if row < 200 or row > 410:
                reward += 10
            if 230 < row < 390:
                reward -= 5

        if done:
            reward = -100
            if not self.have_turned:
                reward -= 100
            if ep_time < 2:
                reward -= 50
        return reward

    def reset(self):
        print('reset')
        self.turn(-1)
        screen = np.array(ImageGrab.grab(bbox=(400, 50, 800, 650)))
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        cnt = 0
        time.sleep(0.5)
        while not self.done(screen) and cnt < 100:
            time.sleep(0.1)
            screen = np.array(ImageGrab.grab(bbox=(400, 50, 800, 650)))
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            cnt += 1
        # pyautogui.click(800, 400)
        PressKey(W)
        time.sleep(0.2)
        ReleaseKey(W)
        time.sleep(1)
        row, col, is_nan = self.process_img(screen)
        state = [row, col, row, col, int(self.mouse_up)]
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
        screen = np.array(ImageGrab.grab(bbox=(400, 50, 800, 650)))
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        done = self.done(screen)
        if not done:
            car_row, car_col, is_nan = self.process_img(screen)
            # print(f'car on {car_row} {car_col}')
            state = [self.prev_state[0], self.prev_state[1], car_row, car_col, int(self.mouse_up)]
            reward = self.get_reward(car_row, car_col)
            self.prev_state = [car_row, car_col]
        else:
            state = [430, 62, 430, 62, 0]
            reward = self.get_reward(430, 62, done=True)
        info = dict()
        info['time'] = time.time() - self.ep_start
        if is_nan:
            if act:
                time.sleep(0.1)
                state, reward, done, info = self.step(action, False)
            else:
                time.sleep(0.2)
                done = True
                state = [430, 62, 430, 62, 0]
                reward = self.get_reward(430.0, 62.0, done=True)
        return state, reward, done, info


if __name__ == '__main__':
    env = Env()
    time.sleep(3)
    pyautogui.click(800, 400)
    time.sleep(0.5)
    env.turn(0.5)
    time.sleep(1.5)
    env.turn(0.01)

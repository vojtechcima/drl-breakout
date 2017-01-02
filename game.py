import numpy as np
import pickle
from ale_python_interface import ALEInterface
from scipy.misc import imresize, imsave

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)


class Game:
    def __init__(self, state_height, state_width, display_screen=False):
        self.ale = ALEInterface()
        self.ale.setInt("frame_skip", 4)
        self.ale.setInt("random_seed", 123)
        self.ale.setBool("display_screen", display_screen)
        self.ale.loadROM("roms/breakout.bin")
        self.actions = self.ale.getMinimalActionSet()
        self.score = 0
        self.actions_len = len(self.actions)
        self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.state_width = state_width
        self.state_height = state_height
        self.state_len = self.state_width * self.state_height
        self.make_move(self.actions[0])
        self.make_move(self.actions[1])

    def get_state(self):
        screen_data = np.zeros(self.screen_width*self.screen_height,dtype=np.uint8)
        self.ale.getScreen(screen_data)
        screen_data_2D = np.reshape(screen_data, (self.screen_height, self.screen_width))
        resized_screen_data_2D = imresize(
            screen_data_2D, (self.state_height, self.state_width))
        resized_screen_data = np.reshape(
            resized_screen_data_2D, self.state_width * self.state_height)
        return resized_screen_data.astype(dtype=np.float32) / 255.0

    def get_state_dims(self):
        return (self.state_width, self.state_height, 1)

    def save_state_to_img(self, fn):
        screen_data = np.zeros(self.screen_width*self.screen_height,dtype=np.uint8)
        self.ale.getScreen(screen_data)
        screen_data_2D = np.reshape(screen_data, (self.screen_height, self.screen_width))
        resized_screen_data_2D = imresize(
            screen_data_2D, (self.state_height, self.state_width))
        imsave(fn, resized_screen_data_2D)
        
    def make_move(self, action):
        r = self.ale.act(action)
        self.score += r
        return r

    def reset_game(self):
        self.ale.reset_game()
        self.score = 0
        self.make_move(self.actions[0])

    def game_over(self):
        return self.ale.game_over()

    def play(self):
        while True: 
            while not self.game_over():
                self.make_move(self.actions[np.random.randint(0, len(self.actions))])
            print("Game Over! Score: %s" % self.score)
            self.reset_game()

    def play_interactive(self):
        """
        play using 0,1,2,3
        save using 8
        """
        buf = []
        while True:
            S = self.get_state()
            a = int(raw_input())
            if(a == 8):
                with open("data.pickle", "w") as f:
                    pickle.dump(buf, f)
                break
            if(a > 3):
                continue
            r = self.make_move(self.actions[a])
            S_ = self.get_state()
            terminal = self.game_over()
            if terminal:
                self.reset_game()
            buf.append((S, a, r, S_, terminal))

if __name__ == "__main__":
    g = Game(84, 84, False)
    g.play_interactive()

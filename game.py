import numpy as np
from ale_python_interface import ALEInterface

class Game:
    def __init__(self, display_screen=False):
        self.ale = ALEInterface()
        self.ale.setInt("frame_skip", 4)
        self.ale.setInt("random_seed", 123)
        self.ale.setBool("display_screen", display_screen)
        self.ale.loadROM("roms/breakout.bin")
        self.actions = self.ale.getMinimalActionSet()
        self.score = 0
        self.actions_len = len(self.actions)
        self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.state_len = self.screen_width * self.screen_height
        self.make_move(self.actions[0])
        self.make_move(self.actions[1])

    def get_state(self):
        screen_data = np.zeros(self.screen_width*self.screen_height,dtype=np.uint8)
        self.ale.getScreen(screen_data)
        return screen_data.astype(dtype=np.float32) / 255.0
        
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
        while not self.game_over():
            self.make_move(self.actions[np.random.randint(0, len(self.actions))])
        print("Game Over! Score: %s" % self.score)
        self.reset_game()


if __name__ == "__main__":
    g = Game(False)
    while True:
        g.play()

import gym
from gym.core import ObservationWrapper
import numpy as np


class PreprocessBreakout(ObservationWrapper):
    @staticmethod
    def _to_grayscale(img):
        """There's a theory that grayscale with emphasized red channel works better"""
        return np.dot(img[..., :3], [0.8, 0.1, 0.1])

    @staticmethod
    def _crop(img, margins=(31, 8, 8, 15)):
        """Remove unnecessary parts of image"""
        return img[margins[0]:-margins[-1], margins[1]:-margins[2]]

    @staticmethod
    def _resize(img):
        """ Simple downsampling to (82, 72)"""
        return img[::2, ::2]

    @staticmethod
    def _to_float(img):
        """More memory to the god of the memory"""
        return np.asarray(img, dtype=np.float64) / 255.0

    def observation(self, img):
        img = self._to_grayscale(img)
        img = self._crop(img)
        img = self._resize(img)
        img = self._to_float(img)
        return img

    def __init__(self, env):
        super(PreprocessBreakout, self).__init__(env)

        self.img_size = (82, 72)  # Eee, magic constants (see crop + downsampling)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(*self.img_size, 1), dtype=np.float64)

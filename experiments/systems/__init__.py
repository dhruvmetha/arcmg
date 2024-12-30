# I want to be able to import the systems from here
from .pendulum import Pendulum
from .cartpole import Cartpole

system_factory = {
    'pendulum': Pendulum,
    'cartpole': Cartpole,
}

__all__ = ['system_factory']
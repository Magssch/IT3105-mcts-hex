import glob
import os

from reinforcement_learner import ReinforcementLearner
from TOPP import TOPP


def clear_models():
    files = glob.glob('models/*')
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    clear_models()

    agent = ReinforcementLearner()
    agent.run()

    topp = TOPP()
    topp.run()

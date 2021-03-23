from reinforcement_learner import ReinforcementLearner
from TOPP import TOPP
import os
import glob

def clear_models():
    files = glob.glob('src/models/*')
    for f in files:
        os.remove(f)

if __name__ == "__main__":
    clear_models()

    agent = ReinforcementLearner()
    agent.run()

    topp = TOPP()
    topp.run()

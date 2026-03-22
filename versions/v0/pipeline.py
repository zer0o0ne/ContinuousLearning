import os
import json

from utils import Logger
from agent.agent import ASI

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    import torch

    # Проверка доступности MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # print(os.path.dirname(__file__))
    log = Logger("frist_exp")
    log(f"Используем: {device}")
    agent = ASI(log)
    agent.test()

if __name__ == "__main__":
    main()
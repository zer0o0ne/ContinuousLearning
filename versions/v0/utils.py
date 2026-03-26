import os
from datetime import datetime

class Logger:
    def __init__(self, base_dir):
        """
        Args:
            base_dir: root directory for this experiment, e.g. data/v0/my_exp
        """
        self.init_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.base_dir = base_dir

        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.filename = os.path.join(logs_dir, f"{self.init_time}.txt")

    def __call__(self, obj):
        text = str(obj)
        print(text)
        with open(self.filename, "a") as f:
            f.write(text + "\n")

    def run_dir(self, scenario_name):
        """Return a timestamped run directory for a training scenario.

        Creates: <base_dir>/<scenario_name>/<init_time>/
        """
        d = os.path.join(self.base_dir, scenario_name, self.init_time)
        os.makedirs(d, exist_ok=True)
        return d

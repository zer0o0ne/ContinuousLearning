import os
from datetime import datetime

class Logger:
    def __init__(self, name, logs_dir = ""):
        init_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.makedirs(f"logs/{name}/{init_time}", exist_ok=True)
        self.filename = f"{logs_dir}/logs/{name}/{init_time}/log.txt"

    def __call__(self, obj):
        text = str(obj)
        print(text)
        with open(self.filename, "a") as f:
            f.write(text + "\n")

    


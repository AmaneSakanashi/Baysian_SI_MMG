import os
import pandas as pd


class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def reset(self, header):
        self.header = header
        self.log = []

    def append(self, l):
        self.log.append(l)

    def get_df(self, steps=None):
        if steps is None:
            log = self.log
        else:
            log = self.log[-steps:]
        #
        df = pd.DataFrame(
            log,
            columns=self.header,
        ).set_index(self.header[0])
        return df

    def to_csv(self, fname, df=None):
        if df is None:
            df = self.get_df()
        path = f"{self.log_dir}/{fname}.csv"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        df.to_csv(path)
        
    def log_pr(self):
        # print(f"{self.log}\n")
        return self.log

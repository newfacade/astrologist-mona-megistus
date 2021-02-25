import json
import sys
import pandas as pd
from datetime import datetime


def compute_frequency(input_file):
    result = {}
    for k, v in json.load(open(input_file)).items():
        for tag, freq in v.items():
            if tag not in result:
                result[tag] = freq
            else:
                result[tag] += freq
    print(result)
    sr = pd.Series(result)
    sr.sort_values(ascending=False, inplace=True)
    print(sr)
    sr.to_excel(datetime.now().strftime("freq-%Y-%m-%d-%H-%M-%S") + ".xlsx", engine='xlsxwriter')
    return result


if __name__ == '__main__':
    compute_frequency(sys.argv[1])

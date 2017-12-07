import sys

with open('results.txt', 'w') as out:
    sys.stdout = out
    print("result")
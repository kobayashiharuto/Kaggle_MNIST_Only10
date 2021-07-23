import pandas as pd

result_csv = pd.read_csv('result/result_9946.csv')
results = result_csv['Label'].to_list()

test_csv = pd.read_csv('result/result.csv')
tests = test_csv['Label'].to_list()

acc = 0
for result, test in zip(results, tests):
    if result == test:
        acc += 1
acc_val = acc / len(results)

print(acc_val)

import pandas as pd

result_csv = pd.read_csv('result/result_9949_v7.csv')
results = result_csv['Label'].to_list()

test_csv = pd.read_csv('result/ensemble_result_4.csv')
tests = test_csv['Label'].to_list()

acc = 0
for result, test in zip(results, tests):
    if result == test:
        acc += 1
acc_val = acc / len(results)

print(acc_val)

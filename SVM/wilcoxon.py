from scipy import stats

knn_path = '/home/osipmd/PycharmProjects/MachineLearning/kNN/knn_result_list.txt'
svm_path = '/home/osipmd/PycharmProjects/MachineLearning/SVM/1/svm_result_list.txt'

with open(knn_path, "r") as ins:
    knn_array = []
    for line in ins:
        knn_array.append(int(line.strip()))

with open(svm_path, "r") as ins:
    svm_array = []
    for line in ins:
        str = line.strip()
        if '-' in str:
            svm_array.append(0)
        else:
            svm_array.append(1)

diff_res = list(map(lambda pair: pair[0] - pair[1], zip(svm_array, knn_array)))
print('diff_res : ', list(filter(lambda val: val != 0, diff_res)))

abs_diff_res = list(map(abs, filter(lambda val: val != 0, diff_res)))

print(abs_diff_res)

not_typical = - sum(diff_res) / abs(sum(diff_res))
if not_typical == -1:
    print('not_typical : svm')
else:
    print('not_typical : knn')

amount_of_not_typical = abs(sum(filter(lambda val: val == not_typical, diff_res)))
print('amount_of_not_typical :', amount_of_not_typical)

Rr = sum(range(1, len(abs_diff_res) + 1)) / len(abs_diff_res)
print('Rr : ', Rr)

T = Rr * amount_of_not_typical
print('T : ', T)
# print(stats.wilcoxon(svm_array, knn_array))
# statistic - The sum of the ranks of the differences above or below zero, whichever is smaller.
# pvalue - The two-sided p-value for the test.

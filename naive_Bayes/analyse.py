import scipy

import statistics
from n_Bayes import train, test

save_ham = False
conflusion_matrix = scipy.array([[0, 0], [0, 0]])
for i in range(10):
    number_of_test_dir = i + 1
    p_spam, p_ham, spam_dict, ham_dict = train(number_of_test_dir)
    matrix = test(number_of_test_dir, p_spam, p_ham, spam_dict, ham_dict, save_ham)
    conflusion_matrix = conflusion_matrix + matrix

f_measure = statistics.Statistics.count_f_measure(conflusion_matrix)

print(conflusion_matrix)
print("F-measure : ", f_measure)

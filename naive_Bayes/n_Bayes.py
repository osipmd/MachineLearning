import math
import os

import scipy

from utils import get_spam_percentage
from utils import is_filename_contains_legit
from utils import is_filename_contains_spmsg
from utils import read_dirs_statistics_except_dir, read_file
from smart_dict import smart_dict


def train(number_of_test_dir):
    spam_count, ham_count, spam_dict, ham_dict = read_dirs_statistics_except_dir(number_of_test_dir)
    p_spam, p_ham, spam_dict, ham_dict = get_probabilities_for_train(spam_count, ham_count, spam_dict, ham_dict)
    return p_spam, p_ham, spam_dict, ham_dict


def test(number_of_test_dir, p_spam, p_ham, spam_dict, ham_dict, save_ham):
    conflusion_matrix = scipy.array([[0, 0], [0, 0]])
    absolute_path = os.getcwd() + '/source_texts/part' + str(number_of_test_dir) + '/'
    for filename in os.listdir(absolute_path):
        dict_from_file = read_file(absolute_path + filename)
        spam_probability, ham_probability = get_probabilities_for_test(dict_from_file, p_spam, p_ham, spam_dict,
                                                                       ham_dict)
        if save_ham == True:
            percentage = 1.0
        else:
            percentage = 0.5
        spam_percentage = get_spam_percentage(spam_probability, ham_probability)
        if spam_percentage >= percentage and is_filename_contains_spmsg(filename):
            conflusion_matrix[0][0] += 1
        elif spam_percentage >= percentage and is_filename_contains_legit(filename):
            conflusion_matrix[0][1] += 1
        elif spam_percentage < percentage and is_filename_contains_spmsg(filename):
            conflusion_matrix[1][0] += 1
        elif spam_percentage < percentage and is_filename_contains_legit(filename):
            conflusion_matrix[1][1] += 1

    return conflusion_matrix


def get_probabilities_for_test(dict_from_file, p_spam, p_ham, spam_dict, ham_dict, coeff_blur=0.001):
    spam_probability = math.log(p_spam)
    ham_probability = math.log(p_ham)

    sum_spam = sum(spam_dict.values())
    sum_ham = sum(ham_dict.values())
    spam_smart_dict = smart_dict(spam_dict)
    ham_smart_dict = smart_dict(ham_dict)

    for key in dict_from_file.keys():
        spam_probability += math.log(dict_from_file[key] * (spam_smart_dict[key] + coeff_blur) /
                                     (sum_spam + 1 * len(spam_smart_dict)))
        ham_probability += math.log(dict_from_file[key] * (ham_smart_dict[key] + coeff_blur) /
                                    (sum_ham + 1 * len(ham_smart_dict)))
    return spam_probability, ham_probability


def get_probabilities_for_train(spam_count, ham_count, spam_dict, ham_dict):
    p_spam = spam_count / (spam_count + ham_count)
    p_ham = ham_count / (spam_count + ham_count)
    return p_spam, p_ham, spam_dict, ham_dict


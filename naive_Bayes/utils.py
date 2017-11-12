import os

import numpy


def read_dirs_statistics_except_dir(except_number_of_dir):
    global_spam_count = 0
    global_ham_count = 0

    global_spam_dict = {}
    global_ham_dict = {}

    for i in range(10):
        if (i + 1) == except_number_of_dir:
            continue
        spam_count, ham_count, spam_dict, ham_dict = read_files_from_directory(i + 1)
        global_spam_count += spam_count
        global_ham_count += ham_count
        global_spam_dict = union_two_dict(global_spam_dict, spam_dict)
        global_ham_dict = union_two_dict(global_ham_dict, ham_dict)

    return global_spam_count, global_ham_count, global_spam_dict, global_ham_dict


def read_files_from_directory(number_of_file, dir_name=os.getcwd() + '/source_texts/part'):
    absolute_path = dir_name + str(number_of_file) + '/'
    spam_count = 0
    ham_count = 0

    spam_dict = {}
    ham_dict = {}

    for filename in os.listdir(absolute_path):
        if is_filename_contains_spmsg(filename):
            spam_count += 1
            spam_dict_from_file = read_file(absolute_path + filename)
            spam_dict = union_two_dict(spam_dict, spam_dict_from_file)
        if is_filename_contains_legit(filename):
            ham_count += 1
            ham_dict_from_file = read_file(absolute_path + filename)
            ham_dict = union_two_dict(ham_dict, ham_dict_from_file)
    return spam_count, ham_count, spam_dict, ham_dict


def read_file(filename):
    i = 0
    dict = {}
    with open(filename) as file:
        for line in file:
            if i == 0:
                dict = parse_subject_line(line)
            if i == 2:
                dict = union_two_dict(dict, parse_line(line))
            i += 1
    return dict


def parse_line(line):
    dict = {}
    words = map(int, line.split())
    for word in words:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1
    return dict


def parse_subject_line(line):
    dict = {}
    split_line = line.split()[1::]
    words = map(int, split_line)
    for word in words:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1
    return dict


def is_filename_contains_spmsg(filename):
    if 'spmsg' in filename:
        return True
    else:
        return False


def is_filename_contains_legit(filename):
    if 'legit' in filename:
        return True
    else:
        return False


def union_two_dict(dict1, dict2):
    union_dict = {k: dict1.get(k, 0) + dict2.get(k, 0)
                  for k in set(dict1) | set(dict2)}
    return union_dict


def get_spam_percentage(spam_probability, ham_probability):
    spam_percentage = 1 / (1 + numpy.exp(ham_probability - spam_probability))
    return spam_percentage

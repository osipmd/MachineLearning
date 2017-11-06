import os


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
        global_spam_dict = {k: global_spam_dict.get(k, 0) + spam_dict.get(k, 0)
                            for k in set(global_spam_dict) | set(spam_dict)}
        global_ham_dict = {k: global_ham_dict.get(k, 0) + ham_dict.get(k, 0)
                           for k in set(global_ham_dict) | set(ham_dict)}

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
            spam_dict = {k: spam_dict.get(k, 0) + spam_dict_from_file.get(k, 0)
                         for k in set(spam_dict) | set(spam_dict_from_file)}
        if is_filename_contains_legit(filename):
            ham_count += 1
            ham_dict_from_file = read_file(absolute_path + filename)
            ham_dict = {k: ham_dict.get(k, 0) + ham_dict_from_file.get(k, 0)
                        for k in set(ham_dict) | set(ham_dict_from_file)}
    return spam_count, ham_count, spam_dict, ham_dict


def read_file(filename):
    i = 0
    with open(filename) as file:
        for line in file:
            if i == 2:
                dict = parse_line(line)
                return dict
            i += 1


def parse_line(line):
    dict = {}
    words = map(int, line.split())
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
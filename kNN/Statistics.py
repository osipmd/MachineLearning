class Statistics:
    @staticmethod
    def count_tp(result, test_data):
        c_func = lambda r, t: r.class_number == 1 and t.class_number == 1
        return sum(map(c_func, result, test_data))

    @staticmethod
    def count_fn(result, test_data):
        count = 0
        for i in range(len(result)):
            if result[i].class_number == 0 and test_data[i].class_number == 1:
                count += 1
        return count

    @staticmethod
    def count_fp(result, test_data):
        count = 0
        for i in range(len(result)):
            if result[i].class_number == 1 and test_data[i].class_number == 0:
                count += 1
        return count

    @staticmethod
    def count_tn(result, test_data):
        count = 0
        for i in range(len(result)):
            if result[i].class_number == 0 and test_data[i].class_number == 0:
                count += 1
        return count

    @staticmethod
    def count_recall(result, test_data):
        tp = Statistics.count_tp(result, test_data)
        fn = Statistics.count_fn(result, test_data)
        if tp == 0:
            print('recall - tp : {}'.format(tp))
            return 0
        return tp / (tp + fn)

    @staticmethod
    def count_precision(result, test_data):
        tp = Statistics.count_tp(result, test_data)
        fp = Statistics.count_fp(result, test_data)
        if tp == 0:
            print('precision - tp : {}'.format(tp))
            return 0
        return tp / (tp + fp)

    @staticmethod
    def count_f_measure(result, test_data, beta=1):
        for point in result:
            print('{} {} {}'.format(point.x, point.y, point.class_number), end=', ')
        print()
        for point in test_data:
            print('{} {} {}'.format(point.x, point.y, point.class_number), end=', ')
        print()
        precision = Statistics.count_precision(result, test_data)
        print('precision : {}'.format(precision))
        recall = Statistics.count_recall(result, test_data)
        print('recall : {}'.format(recall))
        if precision == 0 or recall == 0:
            return 0
        else:
            return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)

class Statistics:
    @staticmethod
    def count_recall(confusion_matrix):
        col_num = len(confusion_matrix)
        total_recall = 0
        for i in range(col_num):
            col_sum = sum(map(lambda row: row[i], confusion_matrix))
            if col_sum == 0:
                recall = 1
            else:
                recall = confusion_matrix[i][i] / col_sum
            total_recall += recall
        return total_recall / col_num

    @staticmethod
    def count_precision(confusion_matrix):
        row_num = len(confusion_matrix)
        total_precision = 0
        for i in range(row_num):
            row_sum = sum(confusion_matrix[i])
            if row_sum == 0:
                precision = 1
            else:
                precision = confusion_matrix[i][i] / row_sum
            total_precision += precision
        return total_precision / row_num

    @staticmethod
    def count_f_measure(confusion_matrix, beta=1):
        precision = Statistics.count_precision(confusion_matrix)
        recall = Statistics.count_recall(confusion_matrix)
        if precision == 0 or recall == 0:
            return 0
        else:
            return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)

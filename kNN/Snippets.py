data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
number_cross_validation = 13
shift = 0

data_size = len(data)
split_size = int(data_size / number_cross_validation)

start = shift * split_size
end = start + split_size

train_data = data[0:start] + data[end:len(data)]
test_data = data[start:end]

print(train_data)
print(test_data)
print(data[10:20])
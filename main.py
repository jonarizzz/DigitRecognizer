import data_converter as dc

TRAIN_FILENAME = "data/train.csv"
TEST_FILENAME = "data/test.csv"

data = dc.csv_data_reader(TRAIN_FILENAME)
dc.detach_labels(data)
data = dc.transform_to_array(data)

print(data[0][0])
print(data[0][1])
print(data[0][2])
print(data[0][3])
print(data[0][4])
print(data[0][5])
print(data[0][6])
print(data[0][7])

data = dc.transform_to_array(dc.csv_data_reader(TEST_FILENAME))

print(data[0][0])
print(data[0][1])
print(data[0][2])
print(data[0][3])
print(data[0][4])
print(data[0][5])
print(data[0][6])
print(data[0][7])

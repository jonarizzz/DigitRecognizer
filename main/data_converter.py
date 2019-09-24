import csv

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"


def csv_data_reader(file_obj):
    with open(file_obj, "r", newline="") as file:
        reader = csv.reader(file)
        rows = list()
        for row in reader:
            rows.append(row)
        return rows


def transform_to_array(data):
    pictures = list()
    picture = list()
    row = list()

    for all_pixels_in_pic in data:
        for i in range(28):
            for j in range(28):
                row.append(all_pixels_in_pic[i * 28 + j])
            picture.append(row)
            row = list()
        pictures.append(picture)

    return pictures


def detach_labels(data):
    for row in data:
        row.pop(0)


# detach pixel names

train_rows = csv_data_reader(TRAIN_FILENAME)
test_rows = csv_data_reader(TEST_FILENAME)

detach_labels(train_rows)
train_data = transform_to_array(train_rows)
test_data = transform_to_array(test_rows)

print(test_data[0][0])
print(train_data[0][0])

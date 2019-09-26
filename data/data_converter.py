import csv


def csv_data_reader(file_name):
    with open(file_name, "r", newline="") as file:
        reader = csv.reader(file)
        rows = list()
        for row in reader:
            rows.append(row)
        rows.pop(0)
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

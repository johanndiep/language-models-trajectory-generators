import csv


def write_string_to_csv(file_path, string_data):
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([string_data])


def read_string_from_csv(file_path):
    with open(file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            return row[0]


if __name__ == "__main__":
    csv_file_path = "frames/plate/robot_command.csv"
    string_to_store = "This is a test command."

    write_string_to_csv(csv_file_path, string_to_store)
    test_string = read_string_from_csv(csv_file_path)

    print(test_string)

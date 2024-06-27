import os
import csv

data_dir = "./test"

def get_labels(root_dir):
    filenames, labels = [], []
    i = 0
    for dir in os.listdir(root_dir):
        # Ignore non-directories
        if not os.path.isdir(os.path.join(root_dir, dir)):
            continue
        
        files = os.listdir(os.path.join(root_dir, dir))
        files.sort(key=lambda x: int(x[6:-4]))

        for file in files:
            filenames.append(file)
            labels.append(i)
        
        i += 1
    
    return filenames, labels

images, labels = get_labels(data_dir)

with open("test_labels.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for image, label in zip(images, labels):
        writer.writerow([image, label])

print('Labels stored successfully!')
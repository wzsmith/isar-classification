import os
import csv
import bisect

DATA_DIR ='data/raw/'
OUT_DIR = 'labels.csv'
class_indexes = [1, 17631, 21510, 25083]

with open(OUT_DIR, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # Get images and sort by index
    images = os.listdir(DATA_DIR)
    images.sort(key=lambda x: int(x[6:-4]) if x.endswith('.png') else 0)

    for image in images:
        if not image.endswith('.png'):
            continue
        
        index = int(image[6:-4])
        # Binary search (more efficient)
        label = bisect.bisect_right(class_indexes, index) - 1  # Find the insertion point (label)
        writer.writerow([image, label])

print(f'Labels stored successfully in {OUT_DIR}!')


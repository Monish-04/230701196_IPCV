import cv2
import os

input_dir = r"C:\Users\palani\Desktop\IPCV\dataset_small"
output_dir = r"C:\Users\palani\Desktop\IPCV\processed_dataset"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for label in os.listdir(input_dir):
    input_path = os.path.join(input_dir, label)
    output_path = os.path.join(output_dir, label)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # 1. Resize
        img = cv2.resize(img, (128, 128))

        # 2. Denoise (Gaussian Blur)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # 3. Enhance (Histogram Equalization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(gray)

        save_path = os.path.join(output_path, img_name)
        cv2.imwrite(save_path, img)

print("Preprocessing DONE")
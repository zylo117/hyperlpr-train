import os

from sklearn.model_selection import train_test_split

IMAGE_PATH = '/home/admin/github/chinese_ocr_keras/datasets/plate'
# TRAIN_IMAGE_PATH = './datasets/license_plate/train'
# VAL_IMAGE_PATH = './datasets/license_plate/val'
LABELS_PATH = './datasets/license_plate/'

pl = os.listdir(IMAGE_PATH)

train_data, val_data = train_test_split(pl, test_size=0.1, random_state=42)

f = open(LABELS_PATH + '/train.txt', 'w')
for d in train_data:
    plate = d.split('_')[0]
    if len(plate) == 7:
        f.write('{}:{}\n'.format(d, plate))
f.close()

f = open(LABELS_PATH + '/val.txt', 'w')
for d in val_data:
    plate = d.split('_')[0]
    if len(plate) == 7:
        f.write('{}:{}\n'.format(d, plate))
f.close()

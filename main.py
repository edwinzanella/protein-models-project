import os
from skimage import imread

# prepare data
input_dir = os.path.join(os.getcwd(), "images")
categories = ['protein', 'not_protein']

data = []
labels = []

for category in categories:
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)

# train/test split

# train classifier

# test performance
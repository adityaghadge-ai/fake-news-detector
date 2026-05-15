import os
import random
import shutil

random.seed(42)
base = 'datasets/images'
classes = ['ai', 'real']

for cls in classes:
    train_dir = os.path.join(base, 'train', cls)
    val_dir = os.path.join(base, 'val', cls)
    files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    random.shuffle(files)
    split = int(len(files) * 0.2)
    for f in files[:split]:
        shutil.move(os.path.join(train_dir, f), os.path.join(val_dir, f))

print('Validation dataset created successfully!')

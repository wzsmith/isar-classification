import imageio
import os

root_dir = './labeled'

for dir in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir)
    img_names = os.listdir(dir_path)
    # Sort images by number
    img_names.sort(key=lambda x: int(x[6:-4]))
    
    images = [imageio.imread(os.path.join(dir_path, img)) for img in img_names[:2880]]

    imageio.mimsave(f'{dir}.gif', images, fps=24)
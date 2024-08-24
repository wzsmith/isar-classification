import imageio.v2 as imageio
import os

root_dir = 'C:/Users/Public/Documents/blender/'
name = 'cubesat'

# dir_path = os.path.join(root_dir, dir)
img_names = os.listdir(root_dir)
img_names = [img for img in img_names if img.endswith('.png')]
# Sort images by number
img_names.sort(key=lambda x: int(x[:4]))

images = [imageio.imread(os.path.join(root_dir, img)) for img in img_names]

imageio.mimsave(f'{name}.gif', images, fps=30)
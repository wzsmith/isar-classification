import cv2
import os

root_dir = 'C:/Users/Public/Documents/processed'
out_dir = 'D:/Users/wzsmith/Documents/Data/CUI2'

for dir in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir)
    if not os.path.isdir(dir_path):
        continue
    img_names = os.listdir(dir_path)
    # Sort images by number
    img_names.sort(key=lambda x: int(x[23:-10]))

    frame = cv2.imread(os.path.join(dir_path, img_names[0]))
    height, width, layers = frame.shape

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, f'{dir}.mp4')
    video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for image in img_names[:3600]:
        video.write(cv2.imread(os.path.join(dir_path, image)))
    
    cv2.destroyAllWindows()
    video.release()
    print(f'Saved {dir}.mp4')



# import imageio
# import os

# root_dir = './labeled'

# for dir in os.listdir(root_dir):
#     dir_path = os.path.join(root_dir, dir)
#     img_names = os.listdir(dir_path)
#     # Sort images by number
#     img_names.sort(key=lambda x: int(x[6:-4]))
    
#     # images = [imageio.imread(os.path.join(dir_path, img)) for img in img_names[:3600]]

#     with imageio.get_writer(f'{dir}.mp4', fps=30) as writer:
#         for img in img_names[:3600]:
#             image = imageio.imread(os.path.join(dir_path, img))

#             writer.append_data(image)


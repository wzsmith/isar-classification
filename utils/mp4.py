import cv2
import os

root_dir = './labeled'

for dir in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir)
    img_names = os.listdir(dir_path)
    # Sort images by number
    img_names.sort(key=lambda x: int(x[6:-4]))

    frame = cv2.imread(os.path.join(dir_path, img_names[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(f'{dir}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for image in img_names[:3600]:
        video.write(cv2.imread(os.path.join(dir_path, image)))
    
    cv2.destroyAllWindows()
    video.release()



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


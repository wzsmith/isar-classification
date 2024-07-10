import gradio as gr
import torch
from torchvision import transforms
from utils.model import CNN_BiLSTM
from PIL import Image
import pandas as pd

# Model Loading (outside of the Gradio interface)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_BiLSTM(num_classes=4)
model.load_state_dict(torch.load("weights/weights200.pth"))
model.to(device)

# Label Map
label_map = {
    0: 'Satellite',
    1: 'Asteroid', 
    2: 'Flat platform',
    3: 'Circular platform'
}

# Function for single image predictions
def predict(img_path):
    transform = transforms.Compose([
        transforms.CenterCrop((120, 120)),
        transforms.ToTensor()
    ])

    # Read in image and pre-process
    image = transform(Image.open(img_path))
    image = image.reshape(1, 1, *image.shape)
    image.to(device)
    
    # Get ground truth label
    labels = pd.read_csv('labels.csv', header=None, names=['image', 'label'])
    img_path = img_path.split('\\')[-1]
    img_label = labels.loc[labels['image'] == img_path, 'label'].squeeze()
    if pd.notnull(img_label):  
        img_label = label_map[img_label]
    else:
        img_label = None

    # Predict label
    with torch.no_grad():
        output = model(image).flatten()
        
    confidences = {label_map[i]: output[i] for i in range(output.shape[0])}
    return confidences, img_label

# Gradio Interface
demo = gr.Interface(
    predict, 
    gr.Image(type='filepath', image_mode='P'), 
    [
        gr.Label(num_top_classes=4, label='prediction'), 
        gr.Label(label='actual label')
    ],
    title='ISAR Image Classifier',
    description='Classifies ISAR images using a CNN-Bi-LSTM model architecture.'
)

demo.launch()
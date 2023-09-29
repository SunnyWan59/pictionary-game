from cnn import CNN
import torch
from PIL import Image
'''
GLOBAL VARIABLES
'''
PATH = "./CNN.pth"
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

# Load the model in 
net  = CNN()
net.load_state_dict(torch.load(PATH))

def resize_image(path,height,width):
    image = Image.open(path)
    new_image = image.resize((height, width))
    new_image.save(path)

def guess(img):
    image = transform(img)
    outputs = net(image)
    _, predicted = torch.max(outputs, 1)
    return predicted

def run():
    torch.multiprocessing.freeze_support()

if __name__ == '__main__':
    run()


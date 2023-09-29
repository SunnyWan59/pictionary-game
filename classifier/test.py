from data_set import *
from cnn import *
PATH = "./CNN.pth"

def run():
    torch.multiprocessing.freeze_support()

if __name__ == '__main__':
    run()
    net = CNN()
    net.load_state_dict(torch.load(PATH))



    correct = 0
    total = 0
    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
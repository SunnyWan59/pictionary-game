from cnn import *
from data_set import *

net = CNN()

lossfunc = nn.CrossEntropyLoss()#multi classification ~
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Function to save the model
def saveModel(name):
    path = f"./{name}.pth"
    torch.save(net.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(test_loader,model):
    
    net.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


def train(model,train_loader, num_epochs,save=False):
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model is on", device)
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = lossfunc(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()     
            if i % 1000 == 999:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        # we want to save the model if the accuracy is the best
        if save:
            saveModel(model.__class__.__name__)
    print("Finished training")

def run():
    torch.multiprocessing.freeze_support()

if __name__ == '__main__':
    run()
    # train(net,training_loader, 20,True)
    PATH = "./CNN.pth"
    net.load_state_dict(torch.load(PATH))
    train(net,training_loader, 40,True)



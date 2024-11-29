import torch
from PIL import Image
from torch import nn, save, load    # Most of the NN classes are here
from torch.optim import Adam        # Optimizer
from torch.utils.data import DataLoader # To load a dataset from PyTorch
from torchvision import datasets
from torchvision.transforms import ToTensor # To convert images to tensors

# Import dataset (MNIST dataset, 0-9)
train = datasets.MNIST(root="data",     # Download here
                       download=True,
                       train=True,      # Train partition
                       transform=ToTensor())    # Data transformations
dataset = DataLoader(train, 32) # 32 Batches
    
# NN Class
class ImageClassifier(nn.Module):   # Subclass it from nn.module class
    def __init__(self):
        super().__init__()  # To subclass this model?
        # Stack layers
        self.model = nn.Sequential(nn.Conv2d(1, 32, (3, 3)),    # 1- B&W, 32- filters of shapes? (kernels), 3x3 shape?
                                   nn.ReLU(),   # Handles non-linearities
                                   nn.Conv2d(32, 64, (3,3)), 
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, (3,3)),    # We shave off 2px off the height and width of each image, adjust?...
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(64*(28-6)*(28-6), 10)    # 1, 28, 28 (mnist img shape), 28-6 since we shave off 2px for each Conv2d layers?..., 10 classes (0-9)
        )
    
    def forward(self, x):  # Call method...
        # pass
        return self.model(x)

# Create an instane of the NN, optimizer, loss
clf = ImageClassifier().to('cuda')  # We'll use GPU, install torch that't CUDA compatible?
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    '''
    #Run training for n #epochs
    for epoch in range(10): #10 epochs
        for batch in dataset:
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')   # Different than in tensorflow (compile and fit)
            yhat = clf(X)   # To generate a prediction
            loss = loss_fn(yhat, y)
            
            #Apply Backpropagation
            opt.zero_grad() # Zero out any existing gradients
            loss.backward()
            opt.step()      # Gradient Descent
    
        print(f"Epoch: {epoch} loss is {loss.item()}")  #Print loss for every batch

    with open('model_state.pt', 'wb') as f: # Save model to our environment, wb- write binary
        save(clf.state_dict(), f)
    '''

    ###
    with open('model_state.pt', 'rb') as f: # Load, read binary
        clf.load_state_dict(load(f))
    
    img = Image.open('2.png')  # Change accordingly
    img = img.convert('L').resize((28, 28)) # Transform input img
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    
    print(torch.argmax(clf(img_tensor)))
    ###
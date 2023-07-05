import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt


#create a linear regression model class
class LinearRegressionModel(nn.Module):#<- almost everthing in pytorch is a nn.Module(think of this as neural network lego blocks)
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(torch.randn(1,#<start with random weights(this will get adjusted as the model learns)
                                              dtype=torch.float),#<-pytorch loves float32 by default
                                  requires_grad=True)#<-can we update this value with gradient descent?)
        self.bias=nn.Parameter(torch.randn(1,#<start with random bias (this will get adjusted as the model learns)
                                           dtype=torch.float),#-<pytorch loves float32 by default
                               requires_grad=True)#<-can we update this value with gradient descent?))


    #forward defines the computation in the model
    def forward(self,x:torch.Tensor)->torch.Tensor: #<-"x" is the input data (e.q. training/testing features)
        return self.weights*x+self.bias #<-this is the linear regression formula (y=mx+b)



# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()


from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model


print(y_preds==loaded_model_preds)

<<<<<<< HEAD
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#create *know* parameters
weight=0.7
bias=0.3

#create data
start=0
end=1
step=0.02

#features
X=torch.arange(start,end,step).unsqueeze(dim=1)
#labels
y=weight*X+bias

print(f'X={X[:10]}')
print(f'y={y[:10]}')


train_spilt=int(0.8*len(X))
x_train,y_train=X[:train_spilt],y[:train_spilt]
x_test,y_test=X[train_spilt:],y[train_spilt:]

print(len(x_train),len(y_train),len(x_test),len(y_test))

def plot_predictions(train_data=x_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_label=y_test,
                     predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c="b",s=4,label="Training data")
    plt.scatter(test_data,test_label,c="g",s=4,label="Testing data")
    if predictions is not None:
        plt.scatter(test_data,predictions,c="r",s=4,label="Predictions")
    plt.legend(prop={"size": 14})
#    plt.show()
    

plot_predictions()

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


#set manual seed since nn.parameter are randomly initialzieed
torch.manual_seed(42)

#create an instance of the model(this isa a subclass of nn.module that contains nn.parameter(s))
model_0=LinearRegressionModel()

#check the nn.parameters within the nn.module subclass we created
list(model_0.parameters())

print(model_0.state_dict())

with torch.inference_mode():
    y_preds=model_0(x_test)

plot_predictions(predictions=y_preds)

print(f"Number of testing samples: {len(x_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

print(f"{y_test-y_preds}")

#create the loss function
loss_fn=nn.L1Loss()#MAE loss is same as L1Loss

#create the optimizer
optimizer=torch.optim.SGD(params=model_0.parameters(), #parameters of target model to optimize
                          lr=0.01) #learning rate(how much the optimizer should change parameters at each step,higher=more(less stable),lower=less(might take a long time))


torch.manual_seed(42)
epochs=1000

train_loss_values=[]
test_loss_values=[]
epoch_count=[]

for epoch in range(epochs):
    ##Training

    #put model in training mode(this is the default state of a model)
    model_0.train()

    #1.forward pass on train data using the forward() method inside
    y_pred=model_0(x_train)
    #print(y_pred)

    #2.calculate the loss (how different are our models predictions to the ground truth)
    loss=loss_fn(y_pred,y_train)

    #3.zero grad of the optimizer
    optimizer.zero_grad()

    #4.loss backwards
    loss.backward()

    #5.progress the optimizer
    optimizer.step()

    #testing

    #put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
        #1.forward pass on test data
        test_pred=model_0(x_test)

        #2.caculate loss on test data
        test_loss=loss_fn(test_pred,y_test.type(torch.float))#predictions come in torch.float datatype

        #print out what's happening
        if epoch%10==0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"epoch:{epoch} | MAE Train loss : {loss} | MAE test loss:{test_loss}")


#plot the loss curves
plt.plot(epoch_count,train_loss_values,label="train loss")
plt.plot(epoch_count,test_loss_values,label="test loss")
plt.title("training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.legend();
plt.show()

print("the model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight},bias: {bias}")


# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(x_test)
print(f"y_preds={y_preds}")

plot_predictions(predictions=y_preds)


from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 


# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(x_test) # perform a forward pass on the test data with the loaded model

print(f"loaded_model_preds={loaded_model_preds}")

print(y_preds==loaded_model_preds)
=======
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#create *know* parameters
weight=0.7
bias=0.3

#create data
start=0
end=1
step=0.02

#features
X=torch.arange(start,end,step).unsqueeze(dim=1)
#labels
y=weight*X+bias

print(f'X={X[:10]}')
print(f'y={y[:10]}')


train_spilt=int(0.8*len(X))
x_train,y_train=X[:train_spilt],y[:train_spilt]
x_test,y_test=X[train_spilt:],y[train_spilt:]

print(len(x_train),len(y_train),len(x_test),len(y_test))

def plot_predictions(train_data=x_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_label=y_test,
                     predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c="b",s=4,label="Training data")
    plt.scatter(test_data,test_label,c="g",s=4,label="Testing data")
    if predictions is not None:
        plt.scatter(test_data,predictions,c="r",s=4,label="Predictions")
    plt.legend(prop={"size": 14})
#    plt.show()
    

plot_predictions()

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


#set manual seed since nn.parameter are randomly initialzieed
torch.manual_seed(42)

#create an instance of the model(this isa a subclass of nn.module that contains nn.parameter(s))
model_0=LinearRegressionModel()

#check the nn.parameters within the nn.module subclass we created
list(model_0.parameters())

print(model_0.state_dict())

with torch.inference_mode():
    y_preds=model_0(x_test)

plot_predictions(predictions=y_preds)

print(f"Number of testing samples: {len(x_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

print(f"{y_test-y_preds}")

#create the loss function
loss_fn=nn.L1Loss()#MAE loss is same as L1Loss

#create the optimizer
optimizer=torch.optim.SGD(params=model_0.parameters(), #parameters of target model to optimize
                          lr=0.01) #learning rate(how much the optimizer should change parameters at each step,higher=more(less stable),lower=less(might take a long time))


torch.manual_seed(42)
epochs=1000

train_loss_values=[]
test_loss_values=[]
epoch_count=[]

for epoch in range(epochs):
    ##Training

    #put model in training mode(this is the default state of a model)
    model_0.train()

    #1.forward pass on train data using the forward() method inside
    y_pred=model_0(x_train)
    #print(y_pred)

    #2.calculate the loss (how different are our models predictions to the ground truth)
    loss=loss_fn(y_pred,y_train)

    #3.zero grad of the optimizer
    optimizer.zero_grad()

    #4.loss backwards
    loss.backward()

    #5.progress the optimizer
    optimizer.step()

    #testing

    #put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
        #1.forward pass on test data
        test_pred=model_0(x_test)

        #2.caculate loss on test data
        test_loss=loss_fn(test_pred,y_test.type(torch.float))#predictions come in torch.float datatype

        #print out what's happening
        if epoch%10==0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"epoch:{epoch} | MAE Train loss : {loss} | MAE test loss:{test_loss}")


#plot the loss curves
plt.plot(epoch_count,train_loss_values,label="train loss")
plt.plot(epoch_count,test_loss_values,label="test loss")
plt.title("training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.legend();
plt.show()

print("the model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight},bias: {bias}")


# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(x_test)
print(f"y_preds={y_preds}")

plot_predictions(predictions=y_preds)


from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 


# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(x_test) # perform a forward pass on the test data with the loaded model

print(f"loaded_model_preds={loaded_model_preds}")

print(y_preds==loaded_model_preds)
>>>>>>> 5279899b69b29cd56fae64d120ae7e49e7589eaf

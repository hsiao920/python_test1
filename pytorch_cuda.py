<<<<<<< HEAD
import torch

torch.cuda.is_available()
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.device_count())


tensor=torch.tensor([1,2,3])
print(tensor,tensor.device)

tensor_on_gpu=tensor.to(device)
print(tensor_on_gpu)

tensor_back_on_cpu=tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
=======
import torch

torch.cuda.is_available()
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.device_count())


tensor=torch.tensor([1,2,3])
print(tensor,tensor.device)

tensor_on_gpu=tensor.to(device)
print(tensor_on_gpu)

tensor_back_on_cpu=tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
>>>>>>> 5279899b69b29cd56fae64d120ae7e49e7589eaf

import torch, numpy, pandas

scalar = torch.tensor(9)

vector = torch.Tensor([9, 9])

matrix = torch.Tensor([[1, 3], [1, 4]])
tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                       [[19, 20, 21], [22, 23, 24], [25, 26, 27]], ])

random_tensor = torch.rand(3, 4)

random_image_size_tensor = torch.rand(size=(224, 224, 3))

zeros = torch.zeros(13, 14)

ones = torch.ones(13, 14)

range_tensor = torch.arange(start=1, end=-900, step=-10)

range_zeros_tensor = torch.zeros_like(range_tensor)
print(range_zeros_tensor)

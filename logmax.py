import torch
import torch.nn as nn

ctc_loss = nn.CTCLoss()
probs = torch.tensor([
    [
        [1.0, 2, 3, 4, 5],
        [3.0, 4, 5, 6, 0],
    ],
])
probs_size = torch.tensor([1, 1])


targets = torch.tensor([
    [1.0, 2, 3, 4, 5],
    [3.0, 4, 5, 6, 0],
])
target_lengths = torch.tensor([5, 5])

probs = probs.log_softmax(1).requires_grad_()

print(probs.shape)
print(targets.shape)
print(probs_size.shape)
print(probs_size)
print(target_lengths.shape)


loss = ctc_loss(probs, targets, probs_size, target_lengths)



print(loss)

# torch.Size([12, 16, 20])
# torch.Size([16, 5])
# torch.Size([16])
# torch.Size([16])

# tensor([12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
#        dtype=torch.int32)
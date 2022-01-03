import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN,self).__init__()

        self.conv_1= nn.Conv2d(1,128, kernel_size=(3,3), padding=(1,1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 8))

        self.linear_1 = nn.Linear(768,64)
        self.linear_2 = nn.Linear(64,20)

        self.GRU= nn.GRU(64,32, bidirectional=True, batch_first=True)

        global counter
        counter =0



    def forward(self,x,targets):
        # x size:[16,1,50,200]
        # targets: [16,5]

        # bs,_,_,_= x.size()
        #
        # x=self.conv_1(x)
        # x= nn.functional.relu(x)
        # x=self.pool_1(x)
        # x = self.conv_2(x)
        # x = nn.functional.relu(x)
        # x = self.pool_2(x)
        #
        # x=x.permute(0,3,1,2)
        # x=x.view(bs, 25, -1)
        #
        # x=self.linear_1(x)
        # x = nn.functional.relu(x)
        #
        # x,h=self.GRU(x)
        #
        # x = self.linear_2(x)
        #
        # x=x.permute(1,0,2)

        global counter
        if counter == 0:
            print(" INPUT SIZE:     ", x.size())

        bs, _, _, _ = x.size()

        x = self.conv_1(x)
        if counter == 0:
            print(" AFTER CONV1:    ", x.size())

        x = nn.functional.relu(x)
        x = self.pool_1(x)

        if counter == 0:
            print(" AFTER POOL1:    ", x.size())

        x = self.conv_2(x)

        if counter == 0:
            print(" AFTER CONV2:    ", x.size())


        x = nn.functional.relu(x)
        x = self.pool_2(x)

        if counter == 0:
            print(" AFTER POOL2:    ", x.size())


        x = x.permute(0, 3, 1, 2)

        if counter == 0:
            print(" AFTER PERMUTE1: ", x.size())
        x = x.view(bs, x.size(1), -1)

        if counter == 0:
            print(" AFTER VIEW:     ", x.size())

        x = self.linear_1(x)

        if counter == 0:
            print(" AFTER LINEAR1:  ", x.size())
        x = nn.functional.relu(x)

        x, h = self.GRU(x)

        if counter == 0:
            print(" AFTER GRU:      ", x.size())

        x = self.linear_2(x)

        if counter == 0:
            print(" AFTER LINEAR2:  ", x.size())

        x = x.permute(1, 0, 2)

        if counter == 0:
            print(" AFTER PERMUTE2:  ", x.size())
            counter= counter+1







        if targets is not None:

            log_probs= nn.functional.log_softmax(x, 2)

            input_lengths=torch.full(size=(bs,), fill_value=log_probs.size(0),
                                     dtype=torch.int32)

            target_lengths = torch.full(size=(bs,), fill_value=targets.size(1),
                                       dtype=torch.int32)

            loss= nn.CTCLoss(blank=19)(log_probs, targets, input_lengths, target_lengths)

            # print(log_probs.shape)
            # print(targets.shape)
            print(input_lengths)
            # print(target_lengths.shape)

            # torch.Size([12, 16, 20])
            # torch.Size([16, 5])
            # torch.Size([16])
            # torch.Size([16])

            return x, loss


        return x,None





































# from torch.utils.data import Dataset, DataLoader
# from skimage import io
#
#
# import torch
#
# '''
# - create Dataset class
# - create  "Create Loaders" function
#     - define trasnforms
#     - split the data
#     - create datasets
#     - create loaders
# '''
#
#
# class CaptchaDataset(Dataset):
#
#     def __init__(self, image_paths, targets, transforms=None):
#
#         self.image_paths=image_paths
#         self.targets=targets
#         self.transform=transforms
#
#
#
#
#     def __getitem__(self, index):
#
#
#         image=io.imread(self.image_paths[index])
#         target= self.targets[index]
#         tensorized_target=torch.tensor(target, dtype=torch.float)
#
#
#         if self.transform:
#             image=self.transform(image)
#
#         return (image,tensorized_target)
#
#
#     def __len__(self):
#         return len(self.image_paths)
#
#
# def Create_Loaders(image_files_paths, targets_enc,batch_size):
#     import torchvision.transforms as transforms
#     from sklearn import model_selection
#
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor()
#
#     ])
#
#     (train_imgs, test_imgs, train_targets, test_targets) = \
#         model_selection.train_test_split(image_files_paths, targets_enc, test_size=0.1, random_state=42)
#
#     train_dataset = CaptchaDataset(image_paths=train_imgs, targets=train_targets, transforms=transform)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
#     test_dataset = CaptchaDataset(image_paths=test_imgs, targets=test_targets, transforms=transform)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, )
#
#
#     return  (train_loader,test_loader)


'''
 - Class for dataset
 - Create Loaders function


'''

from torch.utils.data import Dataset
from skimage import io
import torch
import torchvision.transforms as transforms
from sklearn import model_selection

class CaptchaDataset(Dataset):

    def __init__ (self, image_paths, targets_encoded, transforms= None):
        self.image_paths = image_paths
        self.targets = targets_encoded
        self.transform = transforms

    def __getitem__(self,index):
        image= io.imread(self.image_paths[index])
        target= self.targets[index]
        tensorized_target= torch.tensor(target, dtype=torch.float)

        if self.transform:
            image= self.transform(image)

        return (image, tensorized_target)

    def __len__(self):
        return len(self.image_paths)



def Create_Loaders(image_file_paths, targets_encoded):

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1)  ,
    transforms.ToTensor()

    ])


    (train_imgs, test_imgs, train_targets, test_targets)=model_selection.train_test_split(image_file_paths,targets_encoded, test_size=0.1, random_state=42)

    train_dataset= CaptchaDataset(train_imgs, train_targets, transforms=transform)
    test_dataset= CaptchaDataset(test_imgs, test_targets, transforms=transform)




    train_loader= torch.utils.data.DataLoader(train_dataset, batch_size=16 , shuffle=True)
    test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=16 , shuffle=False)
    # 60.18
    # 60 x16 = 960 , 61th 3
    #
    # 6x16 + 1x11 = 107


    return (train_loader, test_loader)





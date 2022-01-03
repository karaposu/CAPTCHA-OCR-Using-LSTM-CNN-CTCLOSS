from preprocessing import Extract_Data
from dataset import Create_Loaders
from model import CRNN
from TrainEval import train_function, eval_function
from postprocessing import decode_batch_outputs
import torch




def main():
    image_file_paths, targets_encoded, LabelEncoder = Extract_Data()
    train_loader, test_loader = Create_Loaders(image_file_paths, targets_encoded)
    model= CRNN()
    model.to("cuda")
    optimizer= torch.optim.Adam(model.parameters(), lr=3E-4)

    for epoch in range(200):
        Train_Loss= train_function(model, train_loader, optimizer)
        if (epoch+1)%5==0:
            print("Epoch: ", epoch+1, " Loss:",Train_Loss)
            Outputs, Test_Loss =eval_function(model, test_loader, optimizer)
            # Outputs is a list which contains 7 tensor with size of [25,16,20]
            # and each tensor holds the unprocessed information of batch predictions.
            all_predictions=[]
            for e in Outputs:
                batch_predictions=decode_batch_outputs(e,LabelEncoder)
                all_predictions.extend(batch_predictions)

            test_loader_labels=[]
            for images, labels in test_loader:
                for e in labels:
                    # tensor[18., 18., 16., 12., 12.]
                    e=e.type(torch.int).tolist()
                    # [18, 18, 16, 12, 12]
                    test_label_in_characters=LabelEncoder.inverse_transform(e)
                    # ['y', ''y', ' w', 'g', 'g' ]
                    test_label_original=''.join(test_label_in_characters)
                    test_loader_labels.append(test_label_original)





            print(list(zip(test_loader_labels, all_predictions)))
            # from sklearn.metrics import accuracy_score
            # ac=accuracy_score(test_loader_labels, all_predictions)
            # print("Accuracy:", ac)






















if __name__ == "__main__":
            main()
import torch
def decode_batch_outputs(batch_outputs,encoder):
    # [25,16,20] -> [25,1,20]
        # pred= decode ([25,1,20])
    predictions=[]
    for j in range(batch_outputs.shape[1]):
        temp=batch_outputs[:,j,:].unsqueeze(1)
#         [25,20] > [25,1,20]
        prediction=decode_model_output(temp,encoder)
        predictions.append(prediction)

    return predictions



def decode_model_output(model_output, encoder):
    # model_output                             - tensor, shape=[25, 1, 20]
    # model_output_permuted                    - tensor, shape=[1, 25, 20]
    # model_output_converted_to_probabilities  - tensor, shape=[1, 25, 20]
    # model_output_BPA_applied_gpu             - tensor, shape=[1, 25]
    # model_output_BPA_applied                 - numpy   shape=(25,)         [19 19 19 19  14 19 19  14 19 4 19  19 19 16 16 19 4 19 19 19 19 19 19 19 19]
    # model_ouput_label_decoded                - list ,  len= 25             ['_', '_', '_', '_', 'n', '_', '_', 'n', '_', '6', '_', '_', '_', 'w', 'w', '_', '6', '_', '_', '_', '_', '_', '_', '_', '_']
    # model_ouput_without_dublicates           - list ,  len<25              ['_', 'n', '_', 'n', '_', '6', '_', 'w', '_', '6', '_']
    # model_ouput_without_blanks               - list ,  len<25              ['n', 'n', '6', 'w', '6']
    # prediction                               - str                         'nn6w6'

    model_output_permuted=model_output.permute(1,0,2)
    model_output_converted_to_probabilities=torch.softmax(model_output_permuted, 2)
    model_output_BPA_applied_gpu= torch.argmax(model_output_converted_to_probabilities,2)
    model_output_BPA_applied= model_output_BPA_applied_gpu.detach().cpu().numpy().squeeze()

    # Selected Chracters from each timestep:
    # [19 19 19 19  14 19 19  14 19 4 19  19 19 16 16 19 4 19 19 19 19 19 19 19 19]

    # ALPHABET:
    # ['2' '3' '4' '5' '6' '7' '8' 'b' 'c' 'd' 'e' 'f' 'g' 'm' 'n' 'p' 'w' 'x' 'y' '_']

    # Selected Chracters (Alphabet Decoded):
    # ['_', '_', '_', '_', 'n', '_', '_', 'n', '_', '6', '_', '_', '_', 'w', 'w', '_', '6', '_', '_', '_', '_', '_', '_', '_', '_']

    model_ouput_label_decoded=[]
    for n in model_output_BPA_applied:
        if n==19:
            model_ouput_label_decoded.append("_")
        else:
            c=encoder.inverse_transform([n])[0]

            model_ouput_label_decoded.append(c)

    # ['_', '_', '_', '_', 'n', '_', '_', 'n', '_', '6', '_', '_', '_', 'w', 'w', '_', '6', '_', '_', '_', '_', '_', '_', '_', '_']

    model_ouput_without_dublicates=[]
    for i in range(len(model_ouput_label_decoded)):
        if i ==0:
            model_ouput_without_dublicates.append(model_ouput_label_decoded[i])
        else:
            if model_ouput_without_dublicates[-1]!= model_ouput_label_decoded[i]:
                model_ouput_without_dublicates.append(model_ouput_label_decoded[i])

    # ['_', 'n', '_', 'n', '_', '6', '_', 'w', '_', '6', '_']


    model_ouput_without_blanks= []
    for e in model_ouput_without_dublicates:
        if e!="_":
            model_ouput_without_blanks.append(e)

    # ['n', 'n', '6', 'w', '6']

    prediction= "".join(model_ouput_without_blanks)

    return model_ouput_label_decoded





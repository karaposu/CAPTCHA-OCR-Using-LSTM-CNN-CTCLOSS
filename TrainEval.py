# #
#
# DEVICE="cuda"
#
# def train_function(model, train_loader, optimizer):
#
#     model.train()
#     fin_loss = 0
#
#     for data, target in train_loader:
#
#             optimizer.zero_grad()
#             output,loss = model(data.to(DEVICE),target.to(DEVICE))
#             loss.requres_grad = True
#             loss.backward()
#             optimizer.step()
#             fin_loss += loss.item()
#             loss.detach()
#
#
#
#     return fin_loss / len(train_loader)
#
#
# def eval_function(model, data_loader,optimizer):
#     model.eval()
#     fin_loss = 0
#     fin_preds = []
#
#
#     for data, target in data_loader:
#             x=0
#             batch_preds, loss = model(data.to(DEVICE), target.to(DEVICE))
#
#             fin_loss += loss.item()
#             fin_preds.append(batch_preds.detach())
#
#     return fin_preds, fin_loss / len(data_loader)




def train_function(model, train_loader, optimizer):

    model.train()
    final_loss=0


    for data , target in train_loader:
        optimizer.zero_grad()
        output, loss = model(data.to("cuda"), target.to("cuda"))
        loss.requres_grad= True
        loss.backward()
        optimizer.step()
        final_loss+=loss.item()
        loss.detach()

    avr_loss= final_loss/len(train_loader)

    return avr_loss


def eval_function(model, test_loader, optimizer):

    model.eval()
    final_loss=0
    raw_outputs= []


    for data , target in test_loader:
        # optimizer.zero_grad()
        batch_outputs, loss = model(data.to("cuda"), target.to("cuda"))
        loss.requres_grad= True
        # loss.backward()
        # optimizer.step()
        final_loss+=loss.item()

        raw_outputs.append(batch_outputs.detach())

    avr_loss= final_loss/len(test_loader)

    return raw_outputs,avr_loss
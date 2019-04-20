import torch


def test_model(model, test_loader, device, load=False, path=None,
               return_pred=False):

    if load:
        assert(path is not None)
        model.load_state_dict(torch.load(path))

    model.eval()
    model.to(device)

    lbls = []
    pred = []

    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            if return_pred:
                #print(predicted)
                lbls.extend(labels.cpu().tolist())
                pred.extend(predicted.cpu().squeeze().tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {}'.format(
        total, (100 * correct / total)))


    return correct/total, lbls, pred
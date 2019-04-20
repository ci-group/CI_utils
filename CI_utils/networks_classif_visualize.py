import torch
import CI_utils.CI_utils.graphs


def visualize_model(model, train_loader, device):

    model.eval()
    model.to(device)

    lbls = []
    encodings = []

    with torch.no_grad():

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            encoding = model.feature_extractor(inputs)

            lbls.extend(labels.cpu().tolist())
            encodings.extend(encoding.cpu().tolist())

    return encodings, lbls
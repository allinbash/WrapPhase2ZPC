import torch
import h5py


class Predictor:
    def __init__(self, model, device, save_dir):
        self.model = model
        self.device = device
        self.save_dir = save_dir

    def predict(self, dataloader):
        self.model.eval()
        data_save = h5py.File(self.save_dir, 'w')

        with torch.no_grad():
            for t, (X, y) in enumerate(dataloader):
                X = X.to(self.device)
                pred = self.model(X)

                X = X.cpu().numpy().squeeze()
                y = y.squeeze()
                pred = pred.cpu().numpy().squeeze()
                # h5data = data_save.create_group(t)
                # h5data['raw'] = X
                # h5data['label'] = y
                # h5data['pred'] = pred
                print('label:', y)
                print('pred:', pred)


if __name__ == '__main__':
    pass

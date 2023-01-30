import os
import time
import torch.backends.cudnn as cudnn
from utils.losses import *
import utils.data as data
import utils.module as module
import utils.trainer as trainer
import utils.predictor as predictor
import utils.config as config


def main():
    cfg = config.load_config()

    manual_seed = cfg.get('manual_seed', None)
    if manual_seed is not None:
        print('Seed the RNG for all devices with {}'.format(manual_seed))
        torch.manual_seed(manual_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    device = cfg['device']
    base_dir = cfg['base_dir']
    model = module.ZnkCNN(cfg['model']['in_channels'], cfg['model']['out_channels']).to(device)

    if cfg.get('train', True):
        # model.load_state_dict(torch.load('checkpoint/params_maskcnn_MEMS_z5_0.002740.pth'))
        dataloader = data.HdfDataLoader(True, **cfg.get('data'))
        loss_fn = eval(cfg['trainer']['loss'])()
        learning_rate = cfg['trainer']['lr']
        if isinstance(learning_rate, str):
            learning_rate = eval(learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        save_dir = os.path.join(base_dir, 'checkpoint')
        logs_dir = os.path.join(base_dir, 'logs')

        epoch = cfg['trainer']['epoch']
        train_one_epoch = trainer.Trainer(model, loss_fn, optimizer, device, logs_dir)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
        min_loss = 65535

        for t in range(epoch):
            print(' Epoch {} '.format(t).center(64, '-'))
            now_loss = train_one_epoch.train(dataloader)
            scheduler.step()
            if now_loss < min_loss:
                min_loss = now_loss
                torch.save(model, os.path.join(save_dir, 'best.pth'))
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_params.pth'))
            print('Validate Error:\n Avg loss: {:.6f} | Min loss: {:.6f}\n'.format(now_loss, min_loss))

        os.rename(os.path.join(save_dir, 'best.pth'), os.path.join(save_dir, 'best_{:.6f}.pth'.format(min_loss)))
        os.rename(os.path.join(save_dir, 'best_params.pth'), os.path.join(save_dir, 'best_params_{:.6f}.pth'.format(min_loss)))
        print('\nTraining completed!\n')
    else:
        model_dir = os.path.join(base_dir, cfg['predictor']['checkpoint'])
        model.load_state_dict(torch.load(model_dir))
        data_pre = data.HdfDataLoader(False, **cfg.get('data'))
        pred = predictor.Predictor(model, device, os.path.join(base_dir, 'pred.h5'))
        pred.predict(data_pre)
        print('\nPrediction completed!\n')


if __name__ == '__main__':
    time_used = time.time()
    main()
    time_used = time.time() - time_used
    print('Time used: {:.3f} min(s)'.format(time_used / 60))

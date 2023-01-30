import os
import argparse
import yaml
import torch


def load_config():
    parser = argparse.ArgumentParser(description='CNN configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML file')
    parser.add_argument('--train', type=str, default=None, help='"True" for training or "False" for prediction')
    parser.add_argument('--model', type=str, default=None, help='Path to the checkpoint')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    # modify device
    device_name = config.get('device', None)
    if device_name is not None:
        if device_name.startswith('cuda') and not torch.cuda.is_available():
            print('CUDA is not availabel, change to CPU')
            device_name = 'cpu'
    else:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Using {} device'.format(device_name))
    device = torch.device(device_name)
    config['device'] = device

    # modify operating mode
    train_mode = args.train
    if train_mode is not None:
        if train_mode == 'True':
            config['train'] = True
        elif train_mode == 'False':
            config['train'] = False

    # modify dataset path
    base_dir = config.get('base_dir')
    config['data']['training']['path'] = os.path.join(base_dir, config.get('data').get('training').get('path', None))
    config['data']['prediction']['path'] = os.path.join(base_dir, config.get('data').get('prediction').get('path', None))

    # modify checkpoint
    if args.model is not None:
        config['predictor']['checkpoint'] = args.model

    return config

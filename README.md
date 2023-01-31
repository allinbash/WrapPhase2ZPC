# WrapPhase2ZPC
Wrapped phase to Zernike polynomial coefficients.

## Directory Tree
.<br>
├── LICENSE<br>
├── README.md<br>
├── checkpoint<br>
│   ├── best.pth<br>
│   └── best_params.pth<br>
├── config.yml<br>
├── dataset<br>
│   ├── prediction<br>
│   |   └── pr_data.h5<br>
│   └── training<br>
│       └── tr_data.h5<br>
├── logs<br>
├── main.py<br>
├── requirements.txt<br>
└── utils<br>
    ├── config.py<br>
    ├── data.py<br>
    ├── losses.py<br>
    ├── module.py<br>
    ├── predictor.py<br>
    └── trainer.py<br>

## Environment
pip install -r requirements.txt

## Train
python --config config.yml

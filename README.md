# WrapPhase2ZPC
**Wrapped phase to Zernike polynomial coefficients.**

## Directory Tree
```
.
├── LICENSE
├── README.md
├── checkpoint
│   ├── best.pth
│   └── best_params.pth
├── config.yml
├── dataset
│   ├── prediction
│   |   └── pr_data.h5
│   └── training
│       └── tr_data.h5
├── logs
├── main.py
├── requirements.txt
└── utils
    ├── config.py
    ├── data.py
    ├── losses.py
    ├── module.py
    ├── predictor.py
    └── trainer.py
```

## Environment
`pip install -r requirements.txt`

## Train
`python --config config.yml`

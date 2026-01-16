from setuptools import setup, find_packages

setup(
    name = 'DeepGuard',
    version = '0.1.0',
    description = 'DeepFake Detection Model: MultiScaleEfficientViT',
    author = 'seoyunje',
    author_email = "seoyunje2001@gmail.com",
    url = 'https://github.com/HanMoonSub/DeepGuard',

    packages = find_packages(exclude=['Attention', 'preprocess']),
    python_requires = '>=3.8',
    
    # Dependency Library
    install_requires = [
        'torchmetrics',
        'torch >= 1.8.0',
    ],

)
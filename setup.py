import setuptools

with open("README.md", mode='r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'deepguard',
    version = '0.2.1',
    author = 'seoyunje',
    author_email = "seoyunje2001@gmail.com",
    description = "Multi-Scale Efficient Vision Transformer for Robust Deepfake Detection",
    long_description = long_description,
    long_description_content_type ='text/markdown',
    url = 'https://github.com/HanMoonSub/DeepGuard',
    project_urls = {
        "Bug Tracker": "https://github.com/HanMoonSub/DeepGuard/issues",
        "Source Code": "https://github.com/HanMoonSub/DeepGuard",
    },
    classifiers = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent', 
        'Topic :: Scientific/Engineering :: Artificial Intelligence',  
    ],
    packages = setuptools.find_packages(exclude=['Attention', 'preprocess','inference']),
    python_requires = '>=3.10',
    install_requires = []
)
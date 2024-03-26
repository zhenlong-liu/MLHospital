

from setuptools import setup, find_packages

setup(
    name="mlh",
    version="0.1",
    install_requires=[
        'imutils',
        'GPUtil',
        "runx",
        "torchkit",
        "opacus",
        "art",
        "torchvision",
    ],
    packages=["mlh.data_preprocessing",
              "mlh.attacks",
              "mlh.defenses",
              ]
)

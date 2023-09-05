from setuptools import setup

setup(
    name="fourierbasis",
    version="1.0.0",
    description="Encode features in a fourier basis with pytorch.",
    author="Thomas Pönitz",
    author_email="tasptz@gmail.com",
    packages=["fourierbasis"],
    install_requires=["torch"],
)

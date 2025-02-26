from setuptools import setup, find_packages

setup(
    name="cmdpm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.0.0",
        "pyyaml>=6.0.0",
    ],
    python_requires=">=3.8",
)
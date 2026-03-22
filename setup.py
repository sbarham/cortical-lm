from setuptools import setup, find_packages

setup(
    name="cortexlm",
    version="0.1.0",
    description="A Neurophysiologically Structured Language Model",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
        "datasets>=2.0",
        "tokenizers>=0.13",
        "tqdm",
    ],
    extras_require={
        "logging": ["wandb"],
        "tiktoken": ["tiktoken"],
        "dev": ["pytest"],
    },
)

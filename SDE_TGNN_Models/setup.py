"""Setup script for SDE-TGNN package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sde-tgnn",
    version="1.0.0",
    author="SDE-TGNN Authors",
    author_email="sde-tgnn@research.org",
    description=(
        "Stochastic Differential Equation Temporal Graph Neural Network "
        "for Multi-Domain Network Intrusion Detection"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sde-tgnn/sde-tgnn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "flake8>=6.0",
            "black>=23.0",
            "isort>=5.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "sde-tgnn-train=scripts.train:main",
            "sde-tgnn-eval=scripts.evaluate:main",
            "sde-tgnn-preprocess=scripts.preprocess_data:main",
        ],
    },
)

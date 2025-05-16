from setuptools import setup, find_packages

setup(
    name="loan_decisioning_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "Pillow>=10.2.0",
        "pytest>=6.2.5",
    ],
    python_requires=">=3.8",
) 
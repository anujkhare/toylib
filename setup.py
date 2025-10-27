from setuptools import setup, find_packages

setup(
    name="toylib",
    version="0.1.0",
    author="Anuj Khare",
    author_email="khareanuj18@example.com",
    description="A simple library for neural networks in JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/anujkhare/toylib",  # optional
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g.:
        # 'numpy>=1.21.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

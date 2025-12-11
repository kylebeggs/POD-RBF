import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pod_rbf",
    version="2.0.0",
    author="Kyle Beggs",
    author_email="beggskw@gmail.com",
    description="JAX-based POD-RBF for autodiff-enabled reduced order modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kylebeggs/POD-RBF",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

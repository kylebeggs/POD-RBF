import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pod_rbf",  # Replace with your own username
    version="1.1.2",
    author="Kyle Beggs",
    author_email="beggskw@gmail.com",
    description="Tool to perform interpolation using the Proper Orthogonal Decomposition - Radial Basis Function (POD-RBF) method.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UCF-ERAU-OH-Research-Group/POD-RBF",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

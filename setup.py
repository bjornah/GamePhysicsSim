import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GamePhysicsSim", # Replace with your own username
    version="0.1",
    author="Bjorn Ahlgren",
    author_email="bjornah@kth.se",
    description="A package to perform simple physics simulations for objects in games.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bjornah/GamePhysicsSim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
    ],
    python_requires='>=3.5',
)

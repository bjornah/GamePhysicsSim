import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GamePhysicsSim",
    version="0.1",
    author="Bjorn Ahlgren",
    author_email="bjornah@kth.se",
    description="A package to perform simple physics simulations for objects in games.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bjornah/GamePhysicsSim",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: MacOS",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)

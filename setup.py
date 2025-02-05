from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="PlasticEnz",
    version="0.1.0",
    description="A toolset for identifying plastic-degrading enzymes",
    author="Your Name",
    packages=find_packages(include=["PlasticEnz", "PlasticEnz.*"]),
    include_package_data=True,  # Ensure non-code files (like test data) are included
    package_data={"PlasticEnz": ["test/*"]},  # Explicitly include test data
    python_requires=">=3.11.11",
    install_requires=required_packages,
    entry_points={
        "console_scripts": [
            "plasticenz=PlasticEnz.main:main",
        ],
    },
)





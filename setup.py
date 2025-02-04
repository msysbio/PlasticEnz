from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="PlasticEnz",
    version="0.1.0",
    description="A toolset for identifying plastic-degrading enzymes",
    author="Your Name",
    packages=find_packages(),  # Automatically finds all submodules
    include_package_data=True,  # Ensure non-code files are included
    package_data={"": ["test/*"]},  # ✅ Explicitly include test files
    python_requires=">=3.11.11",  # Ensure compatibility with required Python version
    install_requires=required_packages,  # Install dependencies from requirements.txt
    entry_points={
        "console_scripts": [
            "plasticenz=PlasticEnz.main:main",  # Allows running `plasticenz` in terminal
        ],
    },
)


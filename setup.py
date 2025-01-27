from setuptools import setup, find_packages

# Fonction pour lire les dépendances depuis requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]

setup(
    name="omarle",
    version="0.1.0",
    # author="Julien Soule",
    # author_email="julien.soule@lcis.grenoble-inp.fr",
    description="A package containing Atari-like Organizational-oriented MARL environments.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tonnom/omarle",
    packages=find_packages(where='omarle'),
    package_dir={"": "omarle"},
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

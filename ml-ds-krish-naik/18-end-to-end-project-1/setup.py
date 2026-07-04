from setuptools import find_packages, setup

# requirements.txt file contains the list of libraries that our program requires
HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> list[str]:
    """
    This function will return the list of requirements mentioned in the requirements.txt file.
    """
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Muhamamd Umer",
    author_email="ugulzar4512@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

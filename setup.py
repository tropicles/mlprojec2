from setuptools import find_packages,setup
from typing import List

HYPHEN='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path, 'r') as file_obj:
        requirements= file_obj.readlines()
        requirements=[req.replace("/n","") for req in requirements]

        if HYPHEN in requirements:
            requirements.remove(HYPHEN)
    return requirements
    


        




setup(
    name='Mlproject1',
    version='1.0.0',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
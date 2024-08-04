from setuptools import find_packages,setup
from typing import List
HYPHEN_E_DOT = '-e .'


def get_requirements(file_path:str)->List[str]:
   ''' 
   FUNCTION used to get the list of requirements and returned as a list
   '''
   requirements=[]
   with open(file_path) as file_obj:
    requirements=file_obj.readlines()
    # list comprehension
    requirements=[req.replace("\n","") for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    

    return requirements

setup(
name='projectsetup',
version='0.0.1',
author='Thomas Philip',
author_email='talk2tpc@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),

)
from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    HYP_E_DOT = '-e .'
    file_obj = open(file_path,'r')
    text=file_obj.readlines()
    requirements = [req.replace('\n','')  for req in text]
    if HYP_E_DOT in requirements:
        requirements.remove(HYP_E_DOT)
    return requirements



setup(
    name='end to end students performance prediction',
    version='0.0.1',
    author='susil',
    author_email='susilkumarkct@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)
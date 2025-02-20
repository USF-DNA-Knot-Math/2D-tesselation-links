from setuptools import setup, find_packages

setup(
    name='Link_Tesselations_2D',  # The name of your package
    version='0.1',  # Initial version
    packages=find_packages(),  # Automatically find all packages in the directory
    install_requires=['numpy', 'matplotlib'],  # List any dependencies here
    description='A simple package to work with link chord diagram tesselations in 2D',  
    author='Francisco Martinez-Figueroa',  # Your name
    author_email='fmartinezfigueroa@usf.edu',  # Your email
    long_description=open('README.md').read(),  # Read from the README file
    long_description_content_type='text/markdown',  # Markdown format for README
    url='https://github.com/yourusername/mypackage',  # URL of the project
)
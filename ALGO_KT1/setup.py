from setuptools import setup, find_packages
   
classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Operating System :: Microsoft :: Windows :: Windows 11',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
        ]
   
setup(
        name='ALGO_KT1',
        version='0.0.1',
        description='A very basic finance calcs',
        long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
        url='',  
        author='Kiril Tsarvenkov',
        author_email='ktsarvenkov@gmail.com',
        license='MIT', 
        classifiers=classifiers,
        keywords='ALGO', 
        packages=find_packages(),
        install_requires=[''] 
                              )

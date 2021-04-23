import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()    

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
     name='ica_benchmark',  
     version='0.1',
     author="Paulo Augusto Alves Luz Viana",
     author_email="p263889@g.unicamp.br",
     description="Benchmarks for ICA / BSS algorithms",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Kotzly/BCI_MsC",
     packages=setuptools.find_packages(),
    #  packages=["ica_benchmark"],
    #  package_dir={"ica_benchmark": "src"},
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.6',
     install_requires=required,
 )

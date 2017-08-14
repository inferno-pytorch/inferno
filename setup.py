from setuptools import setup, find_packages

# This will be set to true when the release is ready
RELEASED = False

if RELEASED:
    VERSION = '0.0.1'
    LONG_DESCRIPTION = "A utility library and training bench for Pytorch."
    setup_info = dict(
        name='inferno',
        version=VERSION,
        author='Nasim Rahaman',
        author_email='nasim.rahaman@iwr.uni-heidelberg.de',
        url='https://github.com/nasimrahaman/inferno',
        description='Pytorch Training Bench',
        long_description=LONG_DESCRIPTION,
        license='Apache License 2.0',
        packages=find_packages(exclude=('tests',)),
        install_requires=[
            'torch'
        ]
    )
    setup(**setup_info)
else:
    message = "The packaging is not quite ready yet. If you still wish to use inferno on Linux\n" \
              "or OSX, please navigate to the directory where this file is placed and run this\n" \
              "command on a bash shell: `source add2path.sh`"
    print()

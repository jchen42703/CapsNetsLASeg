from setuptools import setup, find_packages

setup(name='capsnets_laseg',
      version='0.01.1',
      description='Capsule Networks for the Automated Segmentation of Left Atrium in Cardiac MRI',
      url='https://github.com/jchen42703/CapsNetsLASeg',
      author='Joseph Chen',
      author_email='jchen42703@gmail.com',
      license='Apache License Version 2.0, January 2004',
      packages=find_packages(),
      install_requires=[
            "numpy>=1.10.2",
            "scipy",
            "scikit-image",
            "future",
            "keras",
            "tensorflow",
            "nibabel",
            "batchgenerators",
            "pandas"
      ],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
          'Intended Audience :: Developers',  # Define that your audience are developers
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: Apache License Version 2.0, January 2004',  # Again, pick a license
          'Programming Language :: Python :: 3',  # Specify which python versions that you want to support
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2.7',
      ],
      keywords=['deep learning', 'image segmentation', 'image classification', 'medical image analysis',
                  'medical image segmentation', 'data augmentation'],
      )

from setuptools import setup, find_packages

setup(name="capsnets_laseg",
      version="0.01.2",
      description="Capsule Networks for the Automated Segmentation of Left Atrium in Cardiac MRI",
      url="https://github.com/jchen42703/CapsNetsLASeg",
      author="Joseph Chen",
      author_email="jchen42703@gmail.com",
      license="Apache License Version 2.0, January 2004",
      packages=find_packages(),
      install_requires=[
            "numpy>=1.10.2",
            "scipy",
            "scikit-image",
            "future",
            "keras==2.3.1",
            "tensorflow==1.14.0",
            "nibabel",
            "batchgenerators==0.19.5",
            "pandas"
      ],
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Developers",  # Define that your audience are developers
          "Topic :: Software Development :: Build Tools",
          "License :: OSI Approved :: Apache License Version 2.0, January 2004",  # Again, pick a license
          "Programming Language :: Python :: 3.6",
      ],
      keywords=["deep learning", "image segmentation", "image classification",
                "medical image analysis", "medical image segmentation",
                "data augmentation", "capsule networks",
                "convolutional neural networks"],
      )

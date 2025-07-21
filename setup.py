from setuptools import setup, find_packages

setup(
    name="phydcm",
    version="1.0.0",
    author="PhyDCM Team",
    author_email="phydcm.team@outlook.com",
    description="An AI-powered DICOM viewer for CT, MRI, and PET images built by medical physics students.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/phydcm/phydcm",
    packages=find_packages(),
    install_requires=[
        "PyQt6>=6.0.0",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "pydicom>=2.4",
        "opencv-python>=4.9",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    project_urls={
        "Documentation": "https://github.com/phydcm/phydcm/wiki",
        "Bug Tracker": "https://github.com/phydcm/phydcm/issues",
    },
)

import setuptools

with open("Readme.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="camai-pkg-castle-guarders",
    version="0.0.1",
    author="Castle Guarders",
    author_email="castleguarders@gmail.com",
    description="AI based Camera monitor and alerting system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/castleguarders/CamAi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Security",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

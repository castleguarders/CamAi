import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    #name="CamAi",
    name="CamAi-castleguarders",
    version="0.0.1",
    author="Castle Guarders",
    author_email="castleguarders@gmail.com",
    description="AI based Camera monitor and alerting system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    scripts=['CamAi/camaicli.py'],
    #package_data={'': ['modeldata/mask_rcnn_coco.h5', 'modeldata/dlib_face_recognition_resnet_model_v1.dat', 'modeldata/shape_predictor_5_face_landmarks.dat', 'modeldata/testimg.jpg']},
    package_data={'': ['example.toml']},
    include_package_data=True,
    install_requires=[],
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

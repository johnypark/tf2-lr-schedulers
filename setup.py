import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf2-lr-schedulers", # Replace with your own username
    version="0.0.1",
    author="John Park",
    author_email="",
    description="TensorFlow2 LR Schedulers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnypark/tf2-lr-schedulers",
    packages=setuptools.find_packages(),
    install_requires = ['tensorflow',
                        'numpy',
                        'tensorflow-addons',
                        'typeguard'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD3 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)


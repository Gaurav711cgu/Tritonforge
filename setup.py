from setuptools import setup, find_packages

setup(
    name="tritonforge",
    version="1.0.0",
    description="Automated GPU kernel optimization and profiling workstation using OpenAI Triton",
    author="Gaurav Kumar Nayak",
    author_email="gauravnayak711@gmail.com",
    url="https://github.com/Gaurav711cgu/TritonForge",
    packages=find_packages(),
    python_requires=">=3.9,<3.12",
    install_requires=[
        "torch>=2.4.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "gpu": [
            "triton>=3.0.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

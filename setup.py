"""
Setup script for Quantitative Trading Platform.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantitative-trading-platform",
    version="1.0.0",
    author="Trading Team",
    author_email="trading@company.com",
    description="High-frequency quantitative trading platform with ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/quantitative-trading-platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.19.0",
            "black>=22.6.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pylint>=2.14.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.4.0",
            "ipykernel>=6.15.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "trading-platform=main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.sql"],
    },
    zip_safe=False,
)

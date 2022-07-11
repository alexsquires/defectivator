from setuptools import setup, find_packages


setup(
    version="0.0.1",
    python_requires=">=3.9",
    license="MIT",
    packages=find_packages(exclude=["docs", "tests*"]),
    name="defectivator",
)

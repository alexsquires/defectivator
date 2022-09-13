from setuptools import setup
import os


SETUP_PTH = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name="defectivator",
        version="0.0.1",
        python_requires=">=3.9",
        license="MIT",
        packages=["defectivator"],
        package_data={"defectivator" : ["data/charges.csv"]},
        url="https://github.com/alexsquires/defectivator"
    )

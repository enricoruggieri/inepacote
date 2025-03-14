from setuptools import setup, find_packages

setup(
    name="inepacote",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas"],
    include_package_data=True,
    package_data={"inepacote": ["synth_model.pkl"]},
    python_requires=">=3.6",
)

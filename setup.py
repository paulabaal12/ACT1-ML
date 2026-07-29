from setuptools import find_packages, setup

setup(
    name="mi_proyecto_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "scikit-learn"],
    description="Paquete de ejemplo con un pipeline de clasificación",
    author="ACT1-ML",
)

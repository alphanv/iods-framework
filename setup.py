from setuptools import setup, find_packages

setup(
    name="iods-framework",
    version="0.1.0",
    description="Intra-Organismal Data Symbiosis: Multimodal Biological Translation",
    author="Alphan Vardarlı",
    author_email="",
    url="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torch-geometric>=2.3",
        "transformers>=4.30",
        "timm>=0.9",
        "numpy>=1.24",
        "scipy>=1.10",
        "biopython>=1.81",
        "librosa>=0.10",
    ],
)

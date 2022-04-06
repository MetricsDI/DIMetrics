import io
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# def get_long_description():
#     readme_file = os.path.join(os.path.dirname(__file__), "README.md")
#     with io.open(readme_file, "r", encoding="utf-8") as f:
#         return f.read()


def get_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with io.open(req_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


requirements = get_requirements()
# long_description = get_long_description()


setup(
    name='torch_lcs_distance',
    version="0.0.0",
    description="PyTorch LCS edit-distance functions",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="",
    # author="",
    # author_email="",
    # license="",
    ext_modules=[
        CUDAExtension('torch_lcs_distance_cuda', [
            'binding.cpp',
            'lcs.cu',
        ])
    ],
    packages=['torch_lcs_distance'],
    cmdclass={
        'build_ext': BuildExtension
    },
    setup_requires=requirements,
    install_requires=requirements
    )

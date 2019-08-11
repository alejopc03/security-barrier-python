from setuptools import setup, find_packages

setup(
    name = "security-barrier",
    version = "0.1",
    packages = find_packages(),

    install_requires = [
        'numpy>=1.12.0',
        'cython'
    ],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['models/**/*', 'labels/**/*']
    },
    include_package_data = True,
    entry_points = {},

    # metadata for upload to PyPI
    author = "Alejandro Pereira Calvo",
    description = "Test port of OpenVINO security barrier demo to Python",
    license = "",
    keywords = "",
    url = ""
)
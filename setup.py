import glob
import os
import subprocess

from setuptools import setup, find_packages

current_file_path = os.path.abspath(os.path.dirname(__file__))


def get_readme():
    readme_file_path = os.path.join(current_file_path, "README.md")
    with open(readme_file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_version():
    version_file_path = os.path.join(current_file_path, "version.txt")
    with open(version_file_path, "r", encoding="utf-8") as f:
        version = f.read().strip()

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:  # pylint: disable=broad-except
        sha = "Unknown"

    if os.getenv("TCC_RELEASE_BUILD") != "1":
        version += ".dev1"
        if sha != "Unknown":
            version += "+" + sha[:7]
    return version


def write_version_python_file(version):
    version_python_file = os.path.join(current_file_path, "text_complexity_computer/version.py")
    with open(version_python_file, "w", encoding="utf-8") as f:
        f.write(f"__version__ = {repr(version)}\n")


def main():
    readme = get_readme()

    version = get_version()
    print("Building version", version)
    write_version_python_file(version)

    packages = find_packages()
    setup(
        name="text_complexity_computer",
        version=version,
        author="le-smog",
        author_email="le-smog@gmx.com",
        license="LGPLv3",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        packages=packages,
        data_files=[("resources.fr", glob.glob('*/*/fr/*')), ("resources.en", glob.glob('*/*/en/*'))],
        include_package_data=True,
        install_requires=["spacy", "numpy", "pandas", "pyahocorasick", "scikit-learn"],
        python_requires=">=3.7",
        description="A library for evaluate text difficulty.",
        long_description=readme,
        long_description_content_type="text/markdown",
    )


if __name__ == "__main__":
    main()

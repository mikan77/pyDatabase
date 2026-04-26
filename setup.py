from pathlib import Path

from setuptools import Extension, find_packages, setup


ROOT = Path(__file__).parent.resolve()


setup(
    name="uspexdb",
    version="0.1.0",
    description="Unified CLI for USPEX database config and contact-query search",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=[
        Extension(
            "uspexdb.query_search._c_anchor",
            sources=[str(ROOT / "src" / "uspexdb" / "query_search" / "_c_anchor.c")],
            extra_compile_args=["-O3"],
        )
    ],
)

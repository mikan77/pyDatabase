from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "uspexdb.query_search._c_anchor",
            sources=["src/uspexdb/query_search/_c_anchor.c"],
            extra_compile_args=["-O3"],
        )
    ],
)

from setuptools import setup

setup(
    extras_require={
        "gpu": [
            "sgl-kernel==0.0.3.post1",
            "sglang[all]==0.4.2",
            "flashinfer"
        ],
    },
    dependency_links=["https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/"],
)

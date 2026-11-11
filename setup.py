from setuptools import setup, find_namespace_packages

setup(
    name="scout",
    version="0.1.0",
    description="SCOUT: an agentic verifier that probes the live environment to judge task completion and safety of computer-use agents",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["desktop_env", "desktop_env.*", "mm_agents", "mm_agents.*", "methods", "methods.scout_verifier", "prompts", "test_time_reflection"]),
    py_modules=["lib_run_single", "lib_results_logger"],
    python_requires="==3.10.*",
    install_requires=[l.strip() for l in open("requirements.txt") if l.strip() and not l.startswith("#")],
)

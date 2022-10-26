import logging
import os
import pathlib
import subprocess

import goldilox
from goldilox.config import CONSTANTS

logger = logging.getLogger()


class DockerFactory:

    def __init__(self, path: str, name: str = "goldilox", image: str = None, dockerfile: str = None, nginx=False):
        self.path = path
        self.name = name
        self.meta = goldilox.Meta.from_file(path)
        self.nginx = nginx
        self.image = image or self._get_image()
        self.dockerfile = dockerfile or self._get_dockerfile()

    @property
    def mlflow(self):
        return os.path.isdir(self.path) and os.path.isfile(os.path.join(self.path, 'MLmodel'))

    @property
    def conda(self):
        return self.meta.env_type == CONSTANTS.CONDA

    def _get_dockerfile(self):
        goldilox_path = pathlib.Path(goldilox.__file__)
        return os.path.join(str(goldilox_path.parent.absolute()), 'app', 'Dockerfile')

    def _get_image(self):
        return 'mambaorg/micromamba' if self.conda else f"python:{self.meta.py_version}-slim-bullseye"

    def get_mlflow_command(self):
        return ['mlflow', 'models', 'build-docker', f"-m", os.path.abspath(self.path), f"-n", self.name,
                "--enable-mlserver"]

    def get_gunicorn_command(self, nginx: bool = False, buildx: bool = None, platform: str = None):
        target_args = self._get_target_args()
        docker_build = ['docker', 'buildx', 'build'] if buildx else ['docker', 'build']

        run_args = docker_build + [f"-f={self.dockerfile}", f"-t={self.name}"]

        build_args = self._get_build_args(nginx=nginx, platform=platform)
        return run_args + target_args + build_args + ['.']

    def _get_target_args(self):
        return ["--target", "conda-image"] if self.conda else ["--target", "venv-image"]

    def _get_build_args(self, nginx=False, platform: str = None):
        version = self.meta.py_version
        if self.conda:
            version = '.'.join(self.meta.py_version.split('.')[:-1])
        build_args = ["--build-arg", f"PYTHON_VERSION={version}",
                      "--build-arg", f"PYTHON_IMAGE={self.image}",
                      "--build-arg", f"GOLDILOX_VERSION={self.meta.goldilox_version}",
                      "--build-arg", f"PIPELINE_FILE={self.path}"]
        if nginx:
            build_args.extend(["--build-arg", f"USE_NGINX=--nginx"])
        if platform is not None:
            build_args = build_args + [f"--platform={platform}"]
        return build_args

    def _get_build_command(self, nginx: bool = False, buildx: bool = None, platform: str = None):
        return self.get_mlflow_command() if self.mlflow else self.get_gunicorn_command(nginx, buildx, platform)

    # TODO remove?
    def build(self, nginx: bool = False, platform: str = None):
        """ build a docker server image"""
        command = self._get_build_command(nginx=nginx, platform=platform)

        logger.info(f"Running docker build as follow:")
        logger.info(f"{' '.join(command)}")
        subprocess.check_call(command)
        platform_str = f"--platform={platform} " if platform is not None else ''
        run_command = f"docker run --rm -it {platform_str}-p 127.0.0.1:5000:5000 {self.name}"
        logger.info(f"Image {self.name} created - run with: '{run_command}'")
        logger.info(f"* On m1 mac you might need to add '-e HOST=0.0.0.0'")
        subprocess.check_call(command)

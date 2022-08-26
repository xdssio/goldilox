import json
import os.path
import shutil
import subprocess
import sys
from pathlib import Path

import click
import cloudpickle

import goldilox
from goldilox import Pipeline
from goldilox.config import CONSTANTS
from goldilox.utils import process_variables, unpickle, get_open


def _read_content(path: str) -> str:
    open_fs = get_open(path)
    with open_fs(path, 'r') as f:
        ret = f.read()
    return ret


def _write_content(output, content):
    with open(output, 'w') as outfile:
        outfile.write(content)
    return output


def process_option(s: str) -> tuple:
    splited = s.split('=')
    if len(splited) != 2:
        splited = s.split(' ')
    if len(splited) == 1:
        key, value = splited[0], True
    else:
        key, value = splited[0], splited[1]
    if key.startswith('--'):
        key = key[2:]
    return key, value


@click.group()
@click.version_option("1.0.0")
def main():
    """This runs every time"""
    pass


@main.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument("path", type=click.Path(exists=True))
@click.argument('options', nargs=-1, type=click.UNPROCESSED)
def serve(path: str, **options):
    """Serve a  pipeline with fastapi server"""
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'MLmodel')):
        command = ['mlflow', 'models', 'serve', f"-m", os.path.abspath(path)] + list(options['options'])
        click.echo(f"Running serve as follow: {' '.join(command)}")
        click.echo(f" ")
        subprocess.check_call(command)
    elif os.path.isdir(path) and os.path.isfile(os.path.join(path, 'pipeline.pkl')) and os.path.isfile(
            os.path.join(path, 'main.py')):
        command = ['gunicorn', 'main:app'] + list(options['options'])
        click.echo(f"Running serve as follow: {' '.join(command)}")
        click.echo(f" ")
        subprocess.check_call(command)

    else:
        server_options = {}

        def clean_key(key):
            if key.startswith('--'):
                key = key[2:]
            elif key.startswith('-'):
                key = key[1:]
            return key.replace('-', '_')

        for option in options['options']:
            splited = option.split('=')
            if len(splited) == 2:
                key, value = splited[0], splited[1]
                server_options[clean_key(key)] = value
            else:
                print(f"(skip) - option {option} was not understood - use key=value version please ")

        from goldilox.app import Server
        if 'bind' not in server_options:
            host = server_options.get('host', os.getenv('HOST', '127.0.0.1'))
            port = server_options.get('port', os.getenv('PORT', 5000))
            server_options['bind'] = f"{host}:{port}"

        Server(path, options=server_options).serve()


@main.command()
def arguments():
    """gunicorn arguments reference"""
    print("check https://docs.gunicorn.org/en/stable/run.html for options")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def description(path: str):
    """print pipeline description"""
    meta = Pipeline.load_meta(path)
    click.echo(meta[CONSTANTS.DESCRIPTION])


@main.command()
@click.argument("path", type=click.Path(exists=True))
def example(path: str):
    """print pipeline output example with all possible outputs"""
    click.echo(json.dumps(Pipeline.load(path).example, indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def raw(path: str):
    """print pipeline input example (raw data)"""
    meta = Pipeline.load_meta(path)
    click.echo(json.dumps(meta.get(CONSTANTS.RAW), indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def variables(path: str):
    """print pipeline variables"""
    meta = Pipeline.load_meta(path)
    click.echo(json.dumps(process_variables(meta.get(CONSTANTS.VARIABLES)), indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def packages(path: str):
    """print pipeline packages"""
    meta = Pipeline.load_meta(path)
    click.echo(json.dumps(meta[CONSTANTS.REQUIREMEMTS], indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option('-o', "--output", type=click.Path())
def freeze(path: str, output: str = None):
    """write pipeline packages to a file (pip freeze > output)"""
    meta = Pipeline.load_meta(path)
    if output is None:
        venv = meta.get(CONSTANTS.VENV_TYPE)
        output = 'environment.yaml' if venv == 'conda' else 'requirements.txt'

    packages = meta.get(CONSTANTS.REQUIREMEMTS)
    _write_content(output, packages)
    click.echo(f"checkout {output}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def meta(path: str):
    """print all meta data"""
    click.echo(json.dumps(Pipeline.load_meta(path), indent=4).replace('\\n', ' '))


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", type=str, default="goldilox")
@click.option('--image', type=str, default=None)
@click.option('--dockerfile', type=click.Path(), default=None)
@click.option('--platform', type=str, default=None)
def build(path: str, name: str = "goldilox", image: str = None, dockerfile: str = None, platform: str = None):
    """ build a docker server image"""
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'MLmodel')):
        command = ['mlflow', 'models', 'build-docker', f"-m", os.path.abspath(path), f"-n", name, "--enable-mlserver"]
    else:
        goldilox_path = Path(goldilox.__file__)
        dockerfile = dockerfile or str(goldilox_path.parent.absolute().joinpath('app').joinpath('Dockerfile'))

        # get meta
        meta = Pipeline.load_meta(path)
        python_version = meta.get(CONSTANTS.PY_VERSION)
        venv_type = meta.get(CONSTANTS.VENV_TYPE)
        goldilox_version = meta.get(CONSTANTS.VERSION)
        is_conda = venv_type == 'conda'
        if image is None:
            image = 'condaforge/mambaforge' if is_conda else f"python:{python_version}-slim-bullseye"
        target_args = ["--target", "conda-image"] if is_conda else ["--target", "venv-image"]

        run_args = ['docker', 'build', f"-f={dockerfile}", f"-t={name}", "--build-arg",
                    f"PYTHON_VERSION={python_version}", "--build-arg", f"PYTHON_IMAGE={image}",
                    "--build-arg", f"GOLDILOX_VERSION={goldilox_version}"]

        build_args = ["--build-arg", f"PIPELINE_FILE={path}"]
        suffix_arg = ['.']
        if platform is not None:
            build_args = build_args + [f"--platform={platform}"]
        command = run_args + target_args + build_args + suffix_arg
    click.echo(f"Running docker build as follow:")
    click.echo(f"{' '.join(command)}")
    click.echo(f" ")
    subprocess.check_call(command)
    platform_str = f"--platform={platform} " if platform is not None else ''
    run_command = f"docker run --rm -it {platform_str}-p 127.0.0.1:5000:5000 {name}"
    click.echo(f"Image {name} created - run with: '{run_command}'")
    click.echo(f"* On m1 mac you might need to add '-e HOST=0.0.0.0'")


@main.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument("path", type=click.Path(exists=True))
@click.argument('key', type=str)
@click.argument('value', type=str)
@click.option('-v', '--variable', is_flag=True)
@click.option('-f', '--file', is_flag=True)
def update(path: str, key: str, value: str, variable: bool, file: bool):
    """
    Update is a tool for updating metadata and variables for a pipeline file.
    Examples:
    $ glx update pipeline.pkl requirements requirements.txt --file # Read the requirements.txt and update the pipeline.pkl file
    $ glx update pipeline.pkl accuracy 0.8 --variable # update the 'accuracy' variable to 0.8 in the pipeline.pkl file
    """

    meta_bytes, state_bytes = Pipeline._read_pipeline_file(path)
    meta = unpickle(meta_bytes)
    tmp_value = value
    if file and not os.path.isfile(value):
        click.echo(f"{value} is not a file - skip")
    if file and os.path.isfile(value):
        tmp_value = _read_content(value)
        click.echo(f"{value} is considered as a file")
    if os.path.isfile(value) and not file:
        click.echo(f"{value} is a file - if you want to load it as such, use the '--file' flag")
        click.echo(f"{value} is considered as a string")
    if variable:
        meta[CONSTANTS.VARIABLES][key] = tmp_value
    elif key in meta:
        meta[key] = tmp_value
    else:
        click.echo(f"{key} was invalid and ignored - for updating a variable use the '--variable' flag")
    click.echo(f"{key} was update to {value}")

    state_to_write = Pipeline.BYTES_SIGNETURE + cloudpickle.dumps(meta) + Pipeline.BYTE_DELIMITER + state_bytes
    Pipeline._save_state(path, state_to_write)


@main.command()
@click.option('-o', '--output', type=str, default=None)
def dockerfile(output: str):
    """Create a Dockerfile for you to work with"""
    goldilox_path = Path(goldilox.__file__)
    docker_file_path = str(goldilox_path.parent.absolute().joinpath('app').joinpath('Dockerfile'))
    docker_output = output or './Dockerfile'
    shutil.copyfile(docker_file_path, docker_output)
    content = _read_content(docker_output)
    click.echo(content)
    click.echo("##################\nDockerfile was writen to './dockerfile'\n")
    click.echo(
        "Run 'docker build -f=Dockerfile -t=<image-name> \
            --build-arg PIPELINE_FILE=<pipeline-path> \
            --build-arg PYTHON_VERSION=<python-version> \
            --build-arg PYTHON_IMAGE=<base-image> \
            --build-arg GOLDILOX_VERSION=<goldilox-version> \
            --target <venv-image/conda-env> .'\
            to build a docker image")


if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("Help stuff")
    main()

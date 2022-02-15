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
from goldilox.config import VARIABLES, RAW, DESCRIPTION, REQUIREMEMTS, PY_VERSION, CONDA_ENV, VERSION
from goldilox.utils import process_variables, unpickle, get_open


@click.group()
@click.version_option("1.0.0")
def main():
    """This runs every time"""
    pass


def process_option(s):
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


@main.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument("path", type=click.Path(exists=True))
@click.argument('options', nargs=-1, type=click.UNPROCESSED)
def serve(path, **options):
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
def description(path):
    """print pipeline description"""
    meta = Pipeline.load_meta(path)
    click.echo(meta[DESCRIPTION])


@main.command()
@click.argument("path", type=click.Path(exists=True))
def example(path):
    """print pipeline output example with all possible outputs"""
    click.echo(json.dumps(Pipeline.load(path).example, indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def raw(path):
    """print pipeline input example (raw data)"""
    meta = Pipeline.load_meta(path)
    click.echo(json.dumps(meta[RAW], indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def variables(path):
    """print pipeline variables"""
    meta = Pipeline.load_meta(path)
    click.echo(json.dumps(process_variables(meta[VARIABLES]), indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def packages(path):
    """print pipeline packages"""
    meta = Pipeline.load_meta(path)
    click.echo(json.dumps(meta[REQUIREMEMTS], indent=4))


def _read_content(path):
    open_fs = get_open(path)
    with open_fs(path, 'r') as f:
        ret = f.read()
    return ret


def _write_content(output, content):
    with open(output, 'w') as outfile:
        outfile.write(content)
    return output


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def freeze(path, output='requirements.txt', ):
    """write pipeline packages to a file (pip freeze > output)"""
    packages = Pipeline.load_meta(path).get(REQUIREMEMTS)
    _write_content(output, packages)
    click.echo(f"checkout {output} for the requirements")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def meta(path):
    """print all meta data"""
    click.echo(json.dumps(Pipeline.load_meta(path), indent=4).replace('\\n', ' '))


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("output", type=str, required=False, default='environment.yml')
def environment(path, output='environment.yml'):
    """write a codna packages to a file (conda env export > output)"""
    env = Pipeline.load_meta(path)
    if env is None:
        raise RuntimeError(f"path: {path} has no valid conda environment in it's metadata")
    env = env[CONDA_ENV]
    _write_content(output, env)
    click.echo(f"checkout {output} for the environment")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", type=str, default="goldilox")
@click.option('--image', type=str, default=None)
@click.option('--dockerfile', type=click.Path(), default=None)
@click.option('--platform', type=str, default=None)
def build(path, name="goldilox", image=None, dockerfile=None, platform=None):
    """ build a docker server image"""
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'MLmodel')):
        command = ['mlflow', 'models', 'build-docker', f"-m", os.path.abspath(path), f"-n", name, "--enable-mlserver"]
    else:
        goldilox_path = Path(goldilox.__file__)
        dockerfile = dockerfile or str(goldilox_path.parent.absolute().joinpath('app').joinpath('Dockerfile'))

        # get meta
        meta = Pipeline.load_meta(path)
        python_version = meta.get(PY_VERSION)
        conda_env = meta.get(CONDA_ENV)
        goldilox_version = meta.get(VERSION)
        if image is None:
            image = 'condaforge/mambaforge' if conda_env is not None else f"python:{python_version}-slim-bullseye"
        target_args = ["--target", "conda-image"] if conda_env is not None else ["--target", "venv-image"]

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
@click.argument('options', nargs=-1, type=click.UNPROCESSED)
def update(path, **options):
    options = options['options']
    meta_bytes, state_bytes = Pipeline._read_pipeline_file(path)
    meta = unpickle(meta_bytes)
    variable_flag = '--variable'
    is_variable = False
    for option in options:
        if option == variable_flag:
            is_variable = True
        else:
            try:
                key_value = option.split('=')
                key, value = key_value[0], key_value[1]
            except:
                click.echo(
                    f"{option} was invalid and ignored - use 'key=value' format")
                continue
            if is_variable:
                meta[VARIABLES][key] = value
                click.echo(f"variable {key} was update to {value}")
            elif key in meta:
                tmp_value = value
                if os.path.isfile(value):
                    click.echo(f"{value} is considered as a file")
                    tmp_value = _read_content(value)
                meta[key] = tmp_value
                click.echo(f"{key} was update to {value}")
            else:
                click.echo(
                    f"{key} was invalid and ignored - for updating a variable use the format '--variable key=value'")
            is_variable = False
    state_to_write = Pipeline.BYTES_SIGNETURE + cloudpickle.dumps(meta) + Pipeline.BYTE_DELIMITER + state_bytes
    Pipeline._save_state(path, state_to_write)


@main.command()
@click.option('--output', type=str, default=None)
def dockerfile(output):
    """Create a Dockerfile for you to work with"""
    goldilox_path = Path(goldilox.__file__)
    docker_file_path = str(goldilox_path.parent.absolute().joinpath('app').joinpath('Dockerfile'))
    docker_output = output or './Dockerfile'
    shutil.copyfile(docker_file_path, docker_output)
    content = _read_content(docker_output)
    click.echo(content)
    click.echo("##################\nDockerfile was writen to './dockerfile'\n")
    click.echo(
        "Use 'docker build -f=Dockerfile -t=<image-name> --build-arg PIPELINE_FILE=<pipeline-path> --build-arg PYTHON_IMAGE=<base-image> .' to build a docker image")


if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("Help stuff")
    main()

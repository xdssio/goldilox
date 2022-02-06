import json
import os.path
import shutil
import subprocess
import sys
from pathlib import Path

import click

import goldilox
from goldilox import Pipeline
from goldilox.config import VARIABLES, RAW, DESCRIPTION, PACKAGES
from goldilox.utils import process_variables


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
        click.echo(f"Running docker build as follow:")
        click.echo(f"{' '.join(command)}")
        click.echo(f" ")
        subprocess.check_call(command)
    else:
        server_options = {}

        def clean_key(key):
            if key.startswith('--'):
                key = key[2:]
            elif key.startswith('-'):
                key = key[1:]
            return key

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
    click.echo(json.dumps(meta[PACKAGES], indent=4))


def _write_content(output, content):
    with open(output, 'w') as outfile:
        outfile.write(content)
    return output


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("output", type=str)
def freeze(path, output='requirements.txt'):
    """write pipeline packages to a file (pip freeze > output)"""
    packages = Pipeline.load_meta(path)[PACKAGES]
    _write_content(output, packages)
    click.echo(f"checkout {output} for the requirements")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", type=str, default="goldilox")
@click.option('--image', type=str, default=None)
@click.option('--platform', type=str, default=None)
def build(path, name="goldilox", image=None, platform=None):
    """ build a docker server image"""
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'MLmodel')):
        command = ['mlflow', 'models', 'build-docker', f"-m", os.path.abspath(path), f"-n", name, "--enable-mlserver"]
    else:
        goldilox_path = Path(goldilox.__file__)
        docker_file_path = str(goldilox_path.parent.absolute().joinpath('app').joinpath('Dockerfile'))
        run_args = ['docker', 'build', f"-f={docker_file_path}", f"-t={name}"]
        build_args = ["--build-arg", f"PIPELINE_FILE={path}"]
        suffix_arg = ['.']
        if image is not None:
            build_args = build_args + ["--build-arg", f"PYTHON_IMAGE={image}"]
        if platform is not None:
            run_args = run_args + [f"--platform={platform}"]
        command = run_args + build_args + suffix_arg
    click.echo(f"Running docker build as follow:")
    click.echo(f"{' '.join(command)}")
    click.echo(f" ")
    subprocess.check_call(command)
    run_command = f"docker run --rm -it {name}" if platform is None else f"docker run --rm -it --platform={platform} -p 127.0.0.1:5000:8000 {name}"
    click.echo(f"Image {name} created - run with: '{run_command}'")


@main.command()
@click.option('--output', type=str, default=None)
def dockerfile(output):
    """Create a Dockerfile for you to work with"""
    goldilox_path = Path(goldilox.__file__)
    docker_file_path = str(goldilox_path.parent.absolute().joinpath('app').joinpath('Dockerfile'))
    docker_output = output or './Dockerfile'
    shutil.copyfile(docker_file_path, docker_output)
    with open(docker_output, 'r') as f:
        click.echo(f.read())
    click.echo("##################\nDockerfile was writen to './dockerfile'\n")
    click.echo(
        "Use 'docker build -f=Dockerfile -t=<image-name> --build-arg PIPELINE_FILE=<pipeline-path> --build-arg PYTHON_IMAGE=<base-image> .' to build a docker image")


if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("Help stuff")
    main()

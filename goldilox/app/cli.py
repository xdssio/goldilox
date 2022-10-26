import json
import os.path
import pathlib
import shutil
import subprocess
import sys

import click
import cloudpickle

import goldilox
import goldilox.app
from goldilox.config import CONSTANTS
from goldilox.utils import process_variables, unpickle


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


def is_mlflow_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, 'MLmodel'))


def is_gunicorn_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, 'pipeline.pkl')) and \
           os.path.isfile(os.path.join(path, 'wsgi.py')) and \
           os.path.isfile(os.path.join(path, 'gunicorn.conf.py'))


def is_ray_dir(path: str) -> bool:
    main_path = os.path.join(path, 'main.py')
    return os.path.isfile(os.path.join(path, 'pipeline.pkl')) and \
           os.path.isfile(main_path) and \
           'ray.init' in pathlib.Path(main_path).read_text()


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("path", type=click.Path(exists=True))
@click.option('--nginx', type=bool, is_flag=True, default=False, help="Make nginx the default when docker run")
@click.option('--nginx-config', type=bool, is_flag=True, default=False, help="Make nginx the default when docker run")
@click.argument("options", nargs=-1, type=click.UNPROCESSED)
def serve(path: str,
          nginx: bool = None,
          nginx_config: str = '',
          **options):
    """Serve a  pipeline with fastapi server"""
    if os.path.isdir(path):
        if is_mlflow_dir(path):
            goldilox.app.MLFlowServer(path=path, options=options).serve()
        elif is_gunicorn_dir(path):
            goldilox.app.GunicornServer(path=path, options=options).serve()
        elif is_ray_dir(path):
            goldilox.app.RayServer(path=path, options=options).serve()
        else:
            click.echo(
                f"A directory was given, but no pipeline was found in it. \nPlease provide the pipeline file directly or provide a 'gunicorn', 'mlflow' or 'ray' directory.\nCheck out")
    else:
        options = options.get('options', [])
        server = goldilox.app.GoldiloxServer(path=path, nginx_config=nginx_config, options=options)
        server.serve(nginx=nginx)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("path", type=click.Path(exists=True))
@click.argument("output", type=str)
@click.option('-f', '--framework', type=click.Choice(['gunicorn', 'mlflow', 'ray'], case_sensitive=False),
              default='gunicorn')
@click.argument('options', nargs=-1, type=click.UNPROCESSED)
def export(path: str, output: str, framework: str, **options):
    """
    Export a pipeline to a directory that can be served with gunicorn, ray or mlflow
    @param path: location of the pipeline
    @param output_path: locaiton of the output directory
    @param framework: framework to use for serving. can be wither 'gunicorn', 'ray' or 'mlflow'. default is 'gunicorn'
    @param nginx: To generate in the case of gunicorn.
    @return:
    """

    if framework == 'mlflow':
        from goldilox.mlops.mlflow import export_mlflow
        export_mlflow(path, output, options)
        click.echo(f"Export to {output} as mlflow")
        click.echo(f" ")
    elif framework == 'gunicorn':
        from goldilox.mlops.gunicorn import export_gunicorn
        export_gunicorn(pipeline=path, path=output)
        click.echo(f"Export to {output} as gunicorn")
    elif framework == 'ray':
        from goldilox.mlops.ray import export_ray
        export_ray(path, output, options)
        click.echo(f"Export to {output} as ray")


def arguments():
    """gunicorn arguments reference"""
    print("check https://docs.gunicorn.org/en/stable/run.html for options")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def description(path: str):
    """print pipeline description"""
    meta = goldilox.Pipeline.load_meta(path)
    click.echo(meta[CONSTANTS.DESCRIPTION])


@main.command()
@click.argument("path", type=click.Path(exists=True))
def example(path: str):
    """print pipeline output example with all possible outputs"""
    click.echo(json.dumps(goldilox.Pipeline.load(path).example, indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def raw(path: str):
    """print pipeline input example (raw data)"""
    meta = goldilox.Pipeline.load_meta(path)
    click.echo(json.dumps(meta.get(CONSTANTS.RAW), indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def variables(path: str):
    """print pipeline variables"""
    meta = goldilox.Pipeline.load_meta(path)
    click.echo(json.dumps(process_variables(meta.get(CONSTANTS.VARIABLES)), indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def packages(path: str):
    """print pipeline packages"""
    meta = goldilox.Pipeline.load_meta(path)
    click.echo(json.dumps(meta[CONSTANTS.REQUIREMEMTS], indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option('-o', "--output", type=click.Path())
def freeze(path: str, output: str = None):
    """write pipeline packages to a file (pip freeze > output)"""
    meta = goldilox.Meta.from_file(path)
    meta.write_environment_file(output)
    click.echo(f"checkout {output}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", type=str, default="goldilox")
@click.option('--image', type=str, default=None)
@click.option('--dockerfile', type=click.Path(), default=None)
@click.option('--nginx', type=bool, is_flag=True, default=False)
@click.option('--buildx', type=bool, is_flag=True, default=False)
@click.option('--platform', type=str, default=None)
def build(path: str, name: str = "goldilox", image: str = None,
          dockerfile: str = None,
          nginx: str = False,
          buildx: str = False,
          platform: str = None):
    """ build a docker server image"""
    import goldilox.app.docker
    factory = goldilox.app.docker.DockerFactory(path, name, image, dockerfile)
    command = factory._get_build_command(nginx=nginx, buildx=buildx, platform=platform)

    click.echo(f"Running docker build as follow:")
    click.echo(f"{' '.join(command)}")
    click.echo(f" ")
    subprocess.check_call(command)
    platform_str = f" --platform={platform}" if platform is not None else ''
    if nginx:
        run_command = f"docker run --rm -it{platform_str} -p 8080:8080 {name}"
    else:
        run_command = f"docker run --rm -it{platform_str} -p 127.0.0.1:5000:5000 {name}"
    click.echo(f"Image {name} created - run with: '{run_command}'")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def meta(path: str):
    """print all meta data"""
    click.echo(json.dumps(goldilox.Meta.from_file(path).to_dict(), indent=4).replace('\\n', ' '))


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

    meta_bytes, state_bytes = goldilox.Pipeline._read_pipeline_file(path)
    meta = unpickle(meta_bytes)
    tmp_value = value
    if file and not os.path.isfile(value):
        click.echo(f"{value} is not a file - skip")
    if file and os.path.isfile(value):
        tmp_value = pathlib.Path(value).read_text()
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

    state_to_write = goldilox.Pipeline.BYTES_SIGNETURE + cloudpickle.dumps(
        meta) + goldilox.Pipeline.BYTE_DELIMITER + state_bytes
    goldilox.Pipeline._save_state(path, state_to_write)


@main.command()
@click.option('-o', '--output', type=str, default=None)
def dockerfile(output: str):
    """Create a Dockerfile for you to work with"""
    goldilox_path = pathlib.Path(goldilox.__file__)
    docker_file_path = str(goldilox_path.parent.absolute().joinpath('app').joinpath('Dockerfile'))
    docker_output = output or './Dockerfile'
    shutil.copyfile(docker_file_path, docker_output)
    content = pathlib.Path(dockerfile).read_text()
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


@main.command()
def hackvaex():
    """Hack vaex version.py for faster loading"""
    click.echo("Overriding vaex versions file")
    import site
    site_packages = site.getsitepackages()[0]
    vaex_version = pathlib.Path(site_packages).joinpath('vaex').joinpath('version.py')
    if vaex_version.exists():
        packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split('\n')
        vaex_versions = [pkg for pkg in packages if 'vaex' in pkg]
        vaex_versions = {pkg.split('==')[0]: pkg.split('==')[1] for pkg in vaex_versions if '==' in pkg}
        file_content = f"""def get_versions():\n\treturn {vaex_versions}"""
        vaex_version.write_text(file_content)


if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("Help stuff")
    main()

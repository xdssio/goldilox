import json
import os.path
import pathlib
import subprocess
import sys

import click
import cloudpickle

import goldilox
import goldilox.app.docker
from goldilox.config import CONSTANTS
from goldilox.utils import process_variables, unpickle, read_text, get_pathlib_path


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
@click.argument("path", type=click.Path())
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
@click.option('-f', '--framework', type=click.Choice(['gunicorn', 'mlflow', 'ray', 'lambda'], case_sensitive=False),
              default='gunicorn')
@click.argument('options', nargs=-1, type=click.UNPROCESSED)
def export(path: str, output: str, framework: str, **options):
    """
    Export a pipeline to a directory that can be served with gunicorn, ray or mlflow
    @param path: location of the pipeline
    @param output_path: locaiton of the output directory
    @param framework: framework to use for serving. can be wither 'gunicorn', 'ray', 'mlflow' or lambda (docker). default is 'gunicorn'
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
        export_ray(path, output)
        click.echo(f"Export to {output} as ray")
    elif framework == 'lambda':
        from goldilox.mlops.aws_lambda import export_lambda
        export_lambda(path, output)
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
@click.argument("path", type=click.Path())
@click.option("--name", type=str, default=None, help="name of the docker image")
@click.option('--image', type=str, default=None,
              help="docker base image to use. default - infer from pipeline environment")
@click.option('--dockerfile', type=click.Path(), default=None, help="path to dockerfile to override the default")
@click.option('--nginx', type=bool, is_flag=True, default=False, help="create a docker which runs nginx as default")
@click.option('--platform', type=str, default=None, help="docker platform to build for")
@click.option('--framework', type=click.Choice(['gunicorn', 'lambda'], case_sensitive=False), default='gunicorn',
              help="framework to use for serving. can be wither 'gunicorn' or 'lambda'. default is 'gunicorn'")
def build(path: str,
          name: str = None,
          image: str = None,
          dockerfile: str = None,
          nginx: str = False,
          platform: str = None,
          framework: str = "gunicorn"):
    """ build a docker server image"""
    if framework == 'lambda':
        factory_class = goldilox.app.docker.LambdaFactory
    else:
        factory_class = goldilox.app.docker.GunicornFactory
    factory = factory_class(path, name, image, dockerfile)
    command = factory._get_build_command(nginx=nginx, platform=platform)

    click.echo(f"Running docker build as follow:")
    click.echo(f"{' '.join(command)}")
    click.echo(f" ")
    subprocess.check_call(command)
    platform_str = f" --platform={platform}" if platform is not None else ''

    if framework == 'lambda':
        run_command = f"docker run -it --rm -p 9000:8080 -v ~/.aws:/root/.aws:ro {factory.name}"
        goldilox_path = pathlib.Path(goldilox.__file__)
        query_file = os.path.join(str(goldilox_path.parent.absolute()), 'mlops', 'aws_lambda', 'query.json')
        with open(query_file, 'r') as f:
            query = json.load(f)
        query['body'] = json.dumps(factory.meta.raw)
        query_command = f"""query.json:\n{json.dumps(query, indent=4)}\ncurl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d @query.json"""
    else:
        run_command = f"docker run --rm -it{platform_str} -p 127.0.0.1:8080:8080 -e WORKERS=1  -v ~/.aws:/root/.aws:ro {factory.name}"
        query_command = f"curl -H 'Content-Type: application/json' -XPOST http://127.0.0.1:8080/inference -d '{json.dumps(factory.meta.raw)}'"

    click.echo(f"Query with\n{query_command}\n")
    click.echo(f"Image {factory.name} created - run with: '{run_command}'")


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
    pathlib = get_pathlib_path(path)
    if file and not pathlib.is_file():
        click.echo(f"{value} is not a file - skip")
    if file and pathlib.is_file():
        tmp_value = pathlib.read_text()
        click.echo(f"{value} is considered as a file")
    if not file and pathlib.is_file():
        click.echo(f"{value} is a file - if you want to load it as such, use the '--file' flag")
        click.echo(f"{value} is considered as a string")
    if variable:
        meta[CONSTANTS.VARIABLES][key] = tmp_value
    elif key in meta:
        meta[key] = tmp_value
    else:
        click.echo(f"{key} was invalid and ignored - for updating a variable use the '--variable' flag")
    state_to_write = CONSTANTS.BYTES_SIGNETURE + cloudpickle.dumps(
        meta) + CONSTANTS.BYTE_DELIMITER + state_bytes
    pathlib.write_bytes(state_to_write)
    click.echo(f"{key} was update to {value}")
    click.echo(f"You can check with 'glx meta {path}'")


@main.command()
@click.option('-o', '--output', type=str, default=None)
def dockerfile(output: str = './Dockerfile'):
    """Create a Dockerfile for you to work with"""
    goldilox_path = get_pathlib_path(goldilox.__file__).parent.absolute()
    docker_file_path = os.path.join(str(goldilox_path), 'app', 'Dockerfile')
    content = read_text(docker_file_path)
    click.echo(content)
    click.echo(f"##################\nDockerfile was writen to '{output}'\n")
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

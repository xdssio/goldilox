import json
import subprocess
import sys

import click

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
    if len(splited) != 2:
        return None, None
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
    """Serve a Goldilox Pipeline"""
    server_options = {}
    for option in options['options']:
        key, value = process_option(option)
        if key is None:
            print(f"option {option} was is illegal and was ignored ")
        else:
            server_options[key] = value
    from goldilox.app import Server
    Server(path, options=server_options).serve()


@main.command()
def arguments():
    print("check https://docs.gunicorn.org/en/stable/run.html for options")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def description(path):
    """Retrive Goldilox Pipeline description"""
    meta = Pipeline.load_meta(path)
    click.echo(meta[DESCRIPTION])


@main.command()
@click.argument("path", type=click.Path(exists=True))
def example(path):
    """Retrive Goldilox Pipeline output example"""
    click.echo(json.dumps(Pipeline.load(path).example, indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def raw(path):
    """Retrive Goldilox Pipeline input example (raw data)"""
    meta = Pipeline.load_meta(path)
    click.echo(json.dumps(meta[RAW], indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def variables(path):
    """Retrive Goldilox Pipeline input example (raw data)"""
    meta = Pipeline.load_meta(path)
    click.echo(json.dumps(process_variables(meta[VARIABLES]), indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def packages(path):
    """Retrive Goldilox Pipeline input example (raw data)"""
    meta = Pipeline.load_meta(path)
    click.echo(json.dumps(meta[PACKAGES], indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("output", type=str)
def freeze(path, output):
    """Retrive Goldilox Pipeline input example (raw data)"""
    packages = Pipeline.load_meta(path)[PACKAGES]
    with open(output, 'w') as outfile:
        outfile.write(packages)
    click.echo(f"checkout {output} for the requirements")


# TODO
@main.command()
@click.argument("path", type=click.Path(exists=True))
def install(path):
    """Install neccecery python packages"""
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           '<packagename>'])

    # process output with an API in the subprocess module:
    reqs = subprocess.check_output([sys.executable, '-m', 'pip',
                                    'freeze']).decode()
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

    print(installed_packages)

    # TODO make freeze to requirements_path
    click.echo(json.dumps(process_variables(Pipeline.load(path).variables), indent=4))


if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("Help stuff")
    main()

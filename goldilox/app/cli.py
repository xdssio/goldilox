import json
import sys

import click
import pandas as pd

from goldilox import Pipeline


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
    from goldilox import Pipeline
    click.echo(Pipeline.load(path).description)


@main.command()
@click.argument("path", type=click.Path(exists=True))
def example(path):
    """Retrive Goldilox Pipeline output example"""
    click.echo(json.dumps(Pipeline.load(path).example, indent=4))


@main.command()
@click.argument("path", type=click.Path(exists=True))
def raw(path):
    """Retrive Goldilox Pipeline input example (raw data)"""
    click.echo(json.dumps(Pipeline.load(path).raw, indent=4))


def process_variables(variables):
    valida_types = {type(None), dict, list, int, float, str, bool}

    return {key: value for key, value in variables.items() if type(value) in valida_types and not pd.isnull(value)}


@main.command()
@click.argument("path", type=click.Path(exists=True))
def variables(path):
    """Retrive Goldilox Pipeline input example (raw data)"""
    click.echo(json.dumps(process_variables(Pipeline.load(path).variables), indent=4))


if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("Help stuff")
    main()

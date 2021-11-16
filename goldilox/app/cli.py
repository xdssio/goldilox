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


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--port", default=5000, help='Server port')
@click.option("--host", default='localhost', help='Server host (default "localhost")')
@click.option("--workers", default=1, help='Number of workers')
def serve(path, port, host, workers, **options):
    """Serve a Goldilox Pipeline"""
    options['bind'] = f"{host}:{port}"
    options['workers'] = workers
    options['preload'] = bool(options.get('--preload', True))

    # click.echo(kwargs)
    from goldilox.app import Server
    Server(path, options=options).serve()


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

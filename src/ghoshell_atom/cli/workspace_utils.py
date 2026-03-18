import click


@click.command(name="init")
def init_workspace():
    raise NotImplementedError("todo")


@click.command(name="env")
def init_env_file_and_check():
    # cp [workspace]/.env.example => [workspace]/.env
    # then checkout the env if minimum valid.
    raise NotImplementedError("todo")


@click.command(name="providers")
def list_providers_from_atom_providers():
    # get Atom instance then print the information of the providers.
    raise NotImplementedError("todo")


@click.command(name="events")
def list_event_models_from_atom(json_schema: bool = False):
    # list the event models of this Atom instance.
    from ghoshell_atom.framework.ghost import Atom
    instance = Atom.get_env_instance()
    models = instance.event_models()
    raise NotImplementedError("todo")

from pathlib import Path

from snk_cli import CLI

roithepig = CLI(Path(__file__).parent.parent)

# @roithepig.app.command()
# def gui():
#     from guigaga.guigaga import GUIGAGA  # lazy load
#     import typer

#     roithepig.app.registered_commands = [c for c in roithepig.app.registered_commands if c.callback.__name__ == "run"]
#     roithepig.app.name = "RoiThePig"
#     GUIGAGA(typer.main.get_command(roithepig.app)).launch()


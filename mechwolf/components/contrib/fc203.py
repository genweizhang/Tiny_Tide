from ..stdlib.active_component import ActiveComponent
from .gsioc import GsiocComponent


class GilsonFC203(ActiveComponent):
    """
    Controls a Gilson FC203B Fraction collector
    """

    metadata = {
        "author": [
            {
                "first_name": "Murat",
                "last_name": "Ozturk",
                "email": "hello@littleblack.fish",
                "institution": "Indiana University, School of Informatics, Computing and Engineering",
                "github_username": "littleblackfish",
            }
        ],
        "stability": "beta",
        "supported": True,
    }

    def __init__(self, name=None, serial_port=None, unit_id=1):
        super().__init__(name=name)

        self.serial_port = serial_port
        self.unit_id = unit_id
        self.position = 1

    def __enter__(self):
        # create the serial connection
        self.gsioc = GsiocComponent(serial_port=self.serial_port, unit_id=self.unit_id)

        self.lock()
        self.gsioc.buffered_command("W1        MechWolf")
        self.gsioc.buffered_command("W2        ")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()

    def lock(self):
        self.gsioc.buffered_command("L0")

    def unlock(self):
        self.gsioc.buffered_command("L1")

    def goto(self, position):
        goto_command = "T" + str(int(position)).zfill(3)
        print(goto_command)
        self.gsioc.buffered_command(goto_command)
        self.gsioc.buffered_command("W2       Collect " + str(position))

    def drain(self):
        drain_command = "Y0000"
        print(drain_command)
        self.gsioc.buffered_command(drain_command)
        self.gsioc.buffered_command("W2         Drain")

    def update(self):
        if self.position == 0:
            self.drain()
        else:
            self.goto(self.position)

    def base_state(self):
        """We assume that the collector starts at the drain position.
        """
        return dict(position=0)

from datetime import datetime


class StdoutPrinter(object):
    def print(self, message):
        print("[+][{}] {}".format(str(datetime.now()), message))


class Console(object):
    LEVEL_INFO = 1
    LEVEL_PROGRESS = 2
    LEVEL_WARNING = 3
    LEVEL_DEBUG = 4

    def __init__(self, printer=StdoutPrinter()):
        self._printer = printer
        self._enabled = {self.LEVEL_INFO, self.LEVEL_PROGRESS, self.LEVEL_WARNING}

    def set_console(self, console):
        self._printer = console

    def _print(self, message, level):
        if level not in self._enabled:
            return

        self._printer.print(message)

    def info(self, message):
        self._print("[INFO    ] " + message, self.LEVEL_INFO)

    def print(self, message):
        self.info(message)

    def progress(self, message):
        self._print("[PROGRESS] " + message, self.LEVEL_PROGRESS)

    def warning(self, message):
        self._print("[WARNING ] " + message, self.LEVEL_WARNING)

    def debug(self, message):
        self._print("[DEBUG   ] " + message, self.LEVEL_DEBUG)

    def _toggle(self, level, state):
        if state:
            self._enabled.add(level)
        else:
            if level in self._enabled:
                self._enabled.remove(level)

    def toggle_info(self, state):
        self._toggle(self.LEVEL_INFO, state)

    def toggle_progress(self, state):
        self._toggle(self.LEVEL_PROGRESS, state)

    def toggle_warning(self, state):
        self._toggle(self.LEVEL_WARNING, state)

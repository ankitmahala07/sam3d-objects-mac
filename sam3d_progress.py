import sys
import time


C = "\033[96m"
RST = "\033[0m"


class CliProgress:
    def __init__(self, total, width=34, initial=0):
        self.total = max(1, int(total))
        self.width = width
        self.done = min(self.total, max(0, int(initial)))
        self.label = "Starting"
        self.started = time.time()
        self.live = sys.stdout.isatty()
        self._rendered = False
        self._last_render = 0.0
        self._stage_steps = {}

    def __call__(self, event, **payload):
        if event == "phase":
            self.label = payload.get("label", self.label)
            self._render(force=True)
            return

        if event == "advance":
            self.label = payload.get("label", self.label)
            self._add(payload.get("amount", 1))
            self._render(force=True)
            return

        if event == "diffusion":
            label = payload.get("label", "Diffusion")
            current = int(payload.get("current", 0))
            total = max(1, int(payload.get("total", 1)))
            previous = self._stage_steps.get(label, 0)
            self._stage_steps[label] = current
            self.label = f"{label} {current}/{total}"
            self._add(max(0, current - previous))
            self._render()

    def advance(self, label, amount=1):
        self("advance", label=label, amount=amount)

    def finish(self, label="Complete"):
        self.done = self.total
        self.label = label
        self._render(force=True)
        self.close()

    def close(self):
        if self.live and self._rendered:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._rendered = False

    def _add(self, amount):
        self.done = min(self.total, self.done + int(amount))

    def _render(self, force=False):
        now = time.time()
        if not force and now - self._last_render < 0.15 and self.done < self.total:
            return
        self._last_render = now

        pct = self.done / self.total
        filled = min(self.width, int(round(self.width * pct)))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = int(now - self.started)
        text = (
            f"  {C}>{RST}  [{bar}] {pct * 100:5.1f}%  "
            f"{self.done}/{self.total}  {self.label}  {elapsed}s"
        )

        if self.live:
            sys.stdout.write("\r" + text + "\033[K")
            sys.stdout.flush()
            self._rendered = True
        elif force:
            print(text, flush=True)

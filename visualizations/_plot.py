"""Matplotlib display helpers."""

import matplotlib.pyplot as plt


def show_or_close(show: bool = True) -> None:
    """Show the figure only when the backend is interactive; otherwise close it."""
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close()

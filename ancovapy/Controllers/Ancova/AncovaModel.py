from ancovapy.Controllers.Ancova.Results import AncovaResults

class AncovaModel:
    """TODO: write docstring for Ancova
    """

    def __init__(self) -> None:
        pass

    def analyze(
        self,
    ) -> AncovaResults:
        """Perform ANCOVA analysis."""
        raise NotImplementedError()

    def plot_residuals(
        self,
    ) -> None:
        """Plot residuals of the ANCOVA model."""
        raise NotImplementedError()

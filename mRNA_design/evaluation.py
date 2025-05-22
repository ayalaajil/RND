
def get_GC_content(rna: str, lower: bool = False) -> float:
    """
    Calculate the GC content of a DNA sequence.

    Args:
        rna (str): The rna sequence.
        lower (bool): If True, converts rna sequence to lowercase before calculation.

    Returns:
        float: The GC content as a percentage.
    """
    if lower:
        rna = rna.lower()
    return (rna.count("G") + rna.count("C")) / len(rna) * 100


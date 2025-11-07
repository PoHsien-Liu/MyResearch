import logging, os

def setup_logger(results_dir: str):
    logger = logging.getLogger("STARE")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # file
    fh = logging.FileHandler(os.path.join(results_dir, "run.log"))
    fh.setFormatter(fmt); logger.addHandler(fh)
    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); logger.addHandler(ch)
    return logger
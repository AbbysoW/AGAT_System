import logging
from datetime import datetime


def setup_logging(level: str = 'INFO'):
    file = logging.FileHandler(
        filename='logger/logs/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log',
        mode='w',
    )
    file.setLevel(logging.DEBUG)

    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)

    logging.basicConfig(
        level= getattr(logging, level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
        handlers=[file, stream]
    )

if __name__ == '__main__':
    setup_logging()

    logger = logging.getLogger(__name__)

    logger.info('test log')
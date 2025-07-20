import datetime
import logging


class Logger:
    """Logger class to log the application process."""
    def __init__(self):
        self._logger = None  # A class variable to hold the logger instance

    @staticmethod
    def get_logger(name=__name__):
        """Static method to return a configured logger."""
        if Logger()._logger is None:
            # Configure the basic settings for the logger
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            logging.basicConfig(
                level=logging.INFO,  # Set the default logging level
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
                handlers=[
                    logging.FileHandler(f"{timestamp}_app.log"),  # Log messages to a file
                    logging.StreamHandler()  # Output log messages to console
                ]
            )

            # Create a logger instance
            Logger._logger = logging.getLogger(name)

        return Logger._logger


logger = Logger.get_logger(__name__)

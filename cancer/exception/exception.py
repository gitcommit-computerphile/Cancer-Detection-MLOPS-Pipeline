import sys


class CancerException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self._format(error_message, error_detail)

    @staticmethod
    def _format(message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            line_no = exc_tb.tb_lineno
            file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            line_no, file_name = "?", "?"
        return f"Error in [{file_name}] line [{line_no}]: {message}"

    def __str__(self):
        return self.error_message
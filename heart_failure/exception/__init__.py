import sys
import traceback
from typing import Optional


def error_message_detail(
    error: Exception,
    error_detail: sys,
    custom_message: Optional[str] = None
) -> str:
    """
    Generate detailed error message with traceback information.

    Args:
        error (Exception): Original exception
        error_detail (sys): sys module
        custom_message (str, optional): Custom error context

    Returns:
        str: Formatted detailed error message
    """

    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    exception_name = error.__class__.__name__

    message = (
        f"\n{'='*80}\n"
        f"❌ Heart Failure Pipeline Exception\n"
        f"{'-'*80}\n"
        f"Exception Type : {exception_name}\n"
        f"File Name      : {file_name}\n"
        f"Line Number    : {line_number}\n"
        f"Error Message  : {str(error)}\n"
    )

    if custom_message:
        message += f"Context        : {custom_message}\n"

    message += f"{'='*80}"

    return message


class HeartFailureException(Exception):

    def __init__(
        self,
        error: Exception,
        error_detail: sys,
        custom_message: Optional[str] = None
    ):
        """
        Args:
            error (Exception): Original raised exception
            error_detail (sys): sys module
            custom_message (str, optional): Extra context
        """
        super().__init__(str(error))

        self.error_message = error_message_detail(
            error=error,
            error_detail=error_detail,
            custom_message=custom_message
        )

    def __str__(self) -> str:
        return self.error_message

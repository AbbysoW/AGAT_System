import re


class Filter: 
    _PATTERN = re.compile(r'[A-Za-z0-9\u0400-\u04FF]')

    def __init__(self):
        pass

    def filter_stt(self, stt_output: str) -> bool:
        fits_pattern = self._fits_pattern(stt_output)
        if not fits_pattern:
            return False
        return True

    def _fits_pattern(self, text: str) -> bool:
        return bool(self._PATTERN.search(text))
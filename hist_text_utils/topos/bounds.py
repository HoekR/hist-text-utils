from dataclasses import dataclass
from typing import List, Tuple
# Assuming fuzzy_search is available in your environment
from fuzzy_search.search.phrase_searcher import PhraseSearcher

@dataclass
class TextSpan:
    label: str
    start: int
    end: int
    metadata: dict = None

class BoundaryDetector:
    def __init__(self, start_patterns: List[str], end_patterns: List[str]):
        self.searcher = PhraseSearcher()
        self.start_patterns = start_patterns
        self.end_patterns = end_patterns

    def detect(self, text: str) -> List[TextSpan]:
        """Scans text for fuzzy anchors and returns logical spans."""
        # 1. Fuzzy search for all start/end markers
        starts = self.searcher.find_matches(text, self.start_patterns)
        ends = self.searcher.find_matches(text, self.end_patterns)
        
        # 2. Logic to pair them (greedy matching or proximity-based)
        spans = self._create_spans(starts, ends)
        return spans

    def _create_spans(self, starts, ends) -> List[TextSpan]:
        # Implementation logic to pair the nearest 'End' to a 'Start'
        spans = []
        for start in starts:
            closest_end = min(ends, key=lambda end: abs(end - start), default=None)
            if closest_end is not None:
                spans.append(TextSpan(label="span", start=start, end=closest_end))
        return spans

import bisect
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PageAnchor:
    page_id: str
    start: int
    end: int

class LinearStream:
    def __init__(self):
        self.text: str = ""
        self.anchors: List[PageAnchor] = []

    def load_page(self, page_id: str, content: str):
        """Stitches a page into the global topos."""
        start_offset = len(self.text)
        self.text += content + "\n"
        end_offset = len(self.text)
        
        self.anchors.append(PageAnchor(page_id, start_offset, end_offset))

    def get_physical_loc(self, global_offset: int) -> Optional[str]:
        """Finds the Page ID for any coordinate in the stream."""
        starts = [a.start for a in self.anchors]
        idx = bisect.bisect_right(starts, global_offset) - 1
        return self.anchors[idx].page_id if idx >= 0 else None
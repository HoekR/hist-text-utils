class RegisterLinker:
    def __init__(self, register: dict):
        """register: { "page_id": ["Name 1", "Name 2"] }"""
        self.register = register
        self.searcher = PhraseSearcher()

    def link_entities(self, stream) -> List[dict]:
        """
        Iterates through the register, finds the page in the stream, 
        and performs a fuzzy search for the names.
        """
        all_hits = []
        for anchor in stream.anchors:
            page_names = self.register.get(anchor.page_id, [])
            if not page_names:
                continue
                
            # Extract only the text for this specific page from the global stream
            page_text = stream.text[anchor.start:anchor.end]
            
            # Find the names within this 'Physical Window'
            matches = self.searcher.find_matches(page_text, page_names)
            
            for m in matches:
                all_hits.append({
                    "name": m.phrase,
                    "global_offset": anchor.start + m.start,
                    "page_id": anchor.page_id
                })
        return all_hits

    def find_in_stream(self, stream, register: dict) -> list:
        hits = []
        for anchor in stream.anchors:
            names = register.get(anchor.page_id, [])
            # Search only the slice of text belonging to this page
            page_text = stream.text[anchor.start : anchor.end]
            # ... fuzzy search logic ...
        return hits

"""
Simple domain-specific spell checker for AdvisorMatch.
"""

class DomainSpellChecker:
    """
    A simple spell checker that can be extended with domain-specific terms.
    For now, it just returns the query as-is (lowercase).
    """
    
    def __init__(self, db_path):
        """Initialize the spell checker with database path."""
        self.db_path = db_path
        # Could load domain-specific terms from database in the future
        
    def correct_text(self, text):
        """
        Correct spelling in the given text.
        Currently just returns lowercased text.
        
        Args:
            text: Input text to correct
            
        Returns:
            Corrected text (currently just lowercased)
        """
        return text.lower().strip()

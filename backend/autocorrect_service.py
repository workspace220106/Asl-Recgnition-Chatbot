from spellchecker import SpellChecker

class AutocorrectService:
    def __init__(self, language='en'):
        self.spell = SpellChecker(language=language)

    def autocorrect(self, text):
        """
        Autocorrects a string of text.
        """
        words = text.split()
        corrected_words = []
        for word in words:
            # Check if word is misspelled
            misspelled = self.spell.unknown([word])
            if misspelled:
                # Get the most likely correction
                correction = self.spell.correction(word)
                if correction:
                    corrected_words.append(correction)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)

# Singleton for English (primary use case for sign letters)
english_autocorrect = AutocorrectService(language='en')

def autocorrect_text(text, lang='en'):
    """
    Main entry point for autocorrecting text.
    Currently optimized for English.
    """
    if lang == 'en':
        return english_autocorrect.autocorrect(text)
    # For other languages, we return the original text 
    # (future: expand SpellChecker support for more languages)
    return text

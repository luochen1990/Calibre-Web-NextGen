"""Publication tiers keep an unproved reading adjacent to its original print.

These choices supply reading assistance, never an identity certificate. Existing
resolved corrections retain their correction notice; undecodable glyphs and
incomplete surviving fragments remain print-only.
"""
import re
import unicodedata
from html import escape

PRINT_ONLY = frozenset({
    'unproved_native_character_map', 'unproved_doubtful_leading_word_ink',
    'incomplete_vertical_word_ink', 'incomplete_numeric_word_ink',
    'unproved_complete_divided_word', 'unproved_complete_word_scope',
})


def typographic_key(value):
    # NFKC expands presentation ligatures but does not strip accents/macrons.
    value = unicodedata.normalize('NFKC', value).replace('\u00ad', '')
    return value.translate(str.maketrans({'’': "'", '‘': "'", 'ʼ': "'",
                                         '“': '"', '”': '"'}))


def equivalent(left, right):
    return typographic_key(left) == typographic_key(right)


def choose(token, decision):
    """Choose assistance from an existing reader, with an explicit provenance.

    Numeric identifiers prefer a source reading when the model dropped a real
    digit. Ordinary disagreements retain the chosen complete first reading;
    a dictionary cannot distinguish two valid words or authorize added letters.
    """
    if decision.get('status') == 'RESOLVED':
        return dict(tier='proved', reading=token, method='existing_resolution')
    method = decision.get('method', '')
    if method in PRINT_ONLY or decision.get('unplaced_prefix'):
        return dict(tier='print', reading=None, method='no_complete_reading')
    source, model = decision.get('source', []), decision.get('model', [])
    if method == 'unproved_mathematical_symbol_identity':
        # A Unicode symbol is a plausible reading. Alphabetic guesses made by
        # both general readers are not evidence of a symbolic font's identity.
        symbolic = next((s for s in source + model
                         if any(unicodedata.category(c).startswith('S') for c in s)), None)
        if symbolic is None:
            return dict(tier='print', reading=None, method='unidentified_symbol')
        return dict(tier='flagged', reading=symbolic, method='reader_unicode_symbol')
    if len(source) == len(model) == 1:
        a, b = source[0], model[0]
        if (any(c.isdigit() for c in a) and any(c.isalpha() for c in a)
                and sum(c.isdigit() for c in a) > sum(c.isdigit() for c in b)
                and len(a) == len(b) + 1
                and any(a[:i] + a[i+1:] == b for i,c in enumerate(a) if c.isdigit())):
            return dict(tier='flagged', reading=a, method='source_complete_numeric_identifier')
    # Private-use, replacement and control glyphs cannot be helpful text. A
    # source-only punctuation fragment is not a plausible whole word either.
    if (not token or not any(c.isalnum() for c in token)
            or any(unicodedata.category(c) in ('Co', 'Cc', 'Cs') or c == '\ufffd' for c in token)
            or token.startswith('<n:')):
        return dict(tier='print', reading=None, method='no_lexical_reading')
    return dict(tier='flagged', reading=token, method='existing_reader_hypothesis')


def prepare(result):
    """Attach publication decisions without erasing the original adjudication."""
    flags = {}
    for index in sorted(result['marked']):
        if result['tokens'][index].startswith('<n:'):
            continue
        old = result.get('word_notes', {}).get(index)
        decision = dict(old or dict(source=[], model=[result['tokens'][index]],
                                    chosen=[result['tokens'][index]], status='UNRESOLVED'))
        choice = choose(result['tokens'][index], decision)
        decision['publication'] = choice
        if choice['tier'] == 'flagged':
            flags[index] = choice
            result['tokens'][index] = choice['reading']
            decision['source_image_only_before_publication'] = decision.get('source_image_only', False)
            decision['source_image_only'] = False
        result.setdefault('word_notes', {})[index] = decision
    result['flagged_readings'] = flags
    return flags


def flagged_html(reading, image, target, style=''):
    """Print evidence is decorative to speech; the live reading is spoken once."""
    metric = ' style="' + escape(style, quote=True) + '"' if style else ''
    return ('<span class="reflow-flagged-reading"><span class="reflow-plausible">'
            + escape(reading) + '</span> '
            + '<a class="source-word-image" href="#' + target + '"><img src="'
            + image + '" alt=""' + metric + '/></a></span>')


def stylesheet():
    return '''\n.reflow-plausible { border-bottom:1px solid currentColor; }
.reflow-flagged-reading { font-style:inherit; }
a.source-word-image, a.source-disclosure-ref { text-decoration:none; color:inherit; }
a.source-word-image img { max-width:none; width:auto; }
a.source-disclosure-ref sup { font-size:.55em; line-height:0; }
'''

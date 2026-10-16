# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-preserving text quality checks, measured on every published page.

These counters never correct spelling or guess a missing word. A rejected layer
must be read from pixels; a rejected reading retains its original source image.
The English dictionary comes from the already installed OCR language data.
"""
from functools import lru_cache
from html.parser import HTMLParser
import hashlib
from pathlib import Path
import re
from shutil import which as _which
import subprocess
from subprocess import run as _run
import tempfile
import unicodedata

VERSION = 'page-text-quality-1'
MIN_PAGE_WORDS = 40
MISS_RATE = .40
MIN_CASE_WORDS = 3
CASE_MISS_RATE = .12
CASE_FLIP_RATE = .08
SEVERE_CASE_TRANSITIONS = 3
WINDOW_WORDS = 16
WINDOW_MISS_RATE = .65
MAX_WORDLIST_BYTES = 8 * 1024 * 1024
_WORDS = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def _latin(word):
    return all(not c.isalpha() or unicodedata.name(c, '').startswith('LATIN ') for c in word)


def _case_anomaly(word):
    # One capital in an otherwise title/camel-case name is ordinary typography.
    # Repeated transitions or an internal uppercase run are damaged casing.
    transitions = sum(a.islower() and z.isupper() for a,z in zip(word,word[1:]))
    return transitions >= 2 or bool(re.search(r'[a-z][A-Z]{2,}',word))


def script_counts(text):
    """Count actual Unicode letter scripts, independent of the English lexicon."""
    result = {}
    for char in text or '':
        if not char.isalpha():
            continue
        name = unicodedata.name(char, '')
        script = name.split(' ', 1)[0]
        if script in {'LATIN', 'GREEK', 'CYRILLIC', 'HEBREW', 'ARABIC', 'DEVANAGARI',
                      'HIRAGANA', 'KATAKANA', 'HANGUL', 'CJK'}:
            result[script] = result.get(script, 0) + 1
    return result


@lru_cache(maxsize=2)
def _load_lexicon(data_path, version, identity):
    """Extract the trusted installed DAWG once, with time and output bounds.

    Identity is the installed language data's SHA; a language upgrade
    cannot reuse a dictionary extracted from different bytes. No PDF text is
    passed to either tool, and no downloaded dictionary or native API is used.
    """
    from . import ocr
    tools = [_which(name) for name in ('combine_tessdata', 'dawg2wordlist')]
    if not all(tools):
        return None
    try:
        data = Path(data_path)
        with tempfile.TemporaryDirectory(prefix='reflow-lexicon-') as directory:
            root = Path(directory)
            chars, dawg, words = (root / name for name in ('eng.lstm-unicharset', 'eng.lstm-word-dawg', 'words.txt'))
            for command in ([tools[0], '-e', str(data), str(chars), str(dawg)],
                            [tools[1], str(chars), str(dawg), str(words)]):
                _run(command, check=True, timeout=20, env=ocr._env(),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if words.stat().st_size > MAX_WORDLIST_BYTES:
                return None
            result = frozenset(word.casefold().replace('’', "'") for word in words.read_text(encoding='utf-8').splitlines())
            return result if len(result) >= 10_000 else None
    except (OSError, subprocess.SubprocessError, UnicodeError, IndexError):
        return None


@lru_cache(maxsize=4)
def _dictionary_identity(executable, stamp, environment):
    # Dictionary availability is independent of the page recognizer's transient
    # failures. Only trusted engine/language files enter this bounded probe.
    listing = _run([executable, '--list-langs'], capture_output=True,
        text=True, check=True, timeout=10, env=dict(environment)).stdout
    match = re.search(r'"([^"\n]+)"', listing.splitlines()[0])
    if not match:
        return None
    data = Path(match[1]) / 'eng.traineddata'
    with data.open('rb') as handle:
        identity = hashlib.file_digest(handle, 'sha256').hexdigest()
    version = _run([executable, '--version'], capture_output=True,
        text=True, check=True, timeout=10, env=dict(environment)).stdout.splitlines()[0]
    return str(data), version, identity, data.stat().st_mtime_ns, data.stat().st_size


def english_lexicon():
    from . import ocr
    executable = _which('tesseract')
    if not executable:
        return None
    try:
        path = Path(executable).resolve()
        stamp = path.stat()
        identity = _dictionary_identity(str(path), (stamp.st_size, stamp.st_mtime_ns),
            tuple(sorted(ocr._env().items())))
        if identity is None:
            return None
        data_path, version, digest, modified, size = identity
        data = Path(data_path).stat()
        if data.st_mtime_ns != modified or data.st_size != size:
            _dictionary_identity.cache_clear()
            return english_lexicon()
        return _load_lexicon(data_path, version, digest)
    except (OSError, subprocess.SubprocessError, UnicodeError, IndexError):
        return None


def inspect(text, *, lexicon=None):
    """Return counters and rejection reasons, never source words or corrections.

    English misses alone need forty words and a forty-percent miss rate. Mixed
    internal case also checks short headings. A sixteen-word sliding window
    catches dense damaged passages that a clean page average would conceal.
    Non-Latin words never count as English dictionary misses.
    """
    lexicon = english_lexicon() if lexicon is None else lexicon
    words = [w for w in _WORDS.findall(text or '') if len(w) >= 3 and _latin(w)]
    folded = [w.casefold().replace('’', "'") for w in words]
    flips = [any(a.islower() and z.isupper() for a, z in zip(w, w[1:])) for w in words]
    anomalous_case_words = sum(_case_anomaly(w) for w in words)
    severe_case_words = sum(sum(a.islower() and z.isupper() for a,z in zip(w,w[1:])) >= SEVERE_CASE_TRANSITIONS for w in words)
    missing = [w not in lexicon for w in folded] if lexicon is not None else []
    count = len(words)
    miss_rate = sum(missing) / count if count and missing else 0.
    flip_rate = sum(flips) / count if count else 0.
    worst = max((sum(missing[n:n+WINDOW_WORDS]) / WINDOW_WORDS
        for n in range(max(0, count-WINDOW_WORDS+1))), default=0.)
    damaged_lines = 0
    if lexicon is not None:
        for line in (text or '').splitlines():
            line_words = [w for w in _WORDS.findall(line) if len(w) >= 3 and _latin(w)]
            if len(line_words) < MIN_CASE_WORDS:
                continue
            line_flips = sum(any(a.islower() and z.isupper() for a, z in zip(w,w[1:])) for w in line_words)
            line_misses = sum(w.casefold().replace('’', "'") not in lexicon for w in line_words)
            if any(_case_anomaly(w) for w in line_words) and line_flips/len(line_words) >= CASE_FLIP_RATE and line_misses/len(line_words) >= CASE_MISS_RATE:
                damaged_lines += 1
    reasons = []
    if lexicon is None and count >= MIN_CASE_WORDS:
        reasons.append('dictionary_unavailable')
    if count >= MIN_PAGE_WORDS and miss_rate >= MISS_RATE:
        reasons.append('dictionary_miss_rate')
    if count >= MIN_CASE_WORDS and anomalous_case_words >= 1 and flip_rate >= CASE_FLIP_RATE and miss_rate >= CASE_MISS_RATE:
        reasons.append('internal_case_and_dictionary_misses')
    if damaged_lines and 'internal_case_and_dictionary_misses' not in reasons:
        reasons.append('internal_case_and_dictionary_misses')
    if severe_case_words:
        reasons.append('repeated_internal_case_transitions')
    if worst >= WINDOW_MISS_RATE:
        reasons.append('dense_dictionary_miss_window')
    return dict(version=VERSION, latin_words=count, dictionary_available=lexicon is not None,
        dictionary_misses=sum(missing), dictionary_miss_rate=round(miss_rate, 6),
        case_flip_words=sum(flips), case_flip_rate=round(flip_rate, 6), anomalous_case_words=anomalous_case_words, severe_case_words=severe_case_words,
        worst_window_miss_rate=round(worst, 6), damaged_case_lines=damaged_lines, scripts=script_counts(text),
        reasons=reasons, needs_read=bool(reasons))


def script_mismatches(source, reading):
    """Reject a reading that loses a source-proven non-Latin passage's script.

    Four source letters avoid treating isolated mathematical variables as prose.
    This is positive Unicode evidence, not a claim that an English recognizer can
    identify every script on a page whose text layer is absent or damaged.
    """
    before, after = script_counts(source), script_counts(reading)
    return [script for script, count in before.items() if script != 'LATIN' and count >= 4
            and after.get(script, 0) < max(2, count * .25)]


def reading_script_mismatches(reading, compiled):
    """An optical fallback cannot replace part of a non-Latin transcription.

    Unlike potentially damaged source-layer evidence, the accepted reader text
    is already the compiler's input. Preserve its complete non-Latin letter
    sequence, allowing Unicode compatibility and case equivalence only.
    """
    def letters(text,script):
        return ''.join(char for char in unicodedata.normalize('NFKC',text).casefold()
            if char.isalpha() and unicodedata.name(char,'').split(' ',1)[0]==script)
    return [script for script,count in script_counts(reading).items()
        if script!='LATIN' and count>=4 and letters(reading,script)!=letters(compiled,script)]


class _ReadingText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        classes = values.get('class', '').split()
        excluded = any(c in {'vision-disclosure', 'source-original', 'source-disclosure', 'source-reading-annotation',
                            'source-evidence-notice', 'source-disclosure-ref', 'note-return'} for c in classes)
        if tag not in {'br', 'img', 'hr', 'meta', 'link', 'input'}:
            self.stack.append((tag, excluded or any(v for _, v in self.stack)))
        if tag in {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'aside', 'br', 'li', 'tr', 'td', 'th'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        for n in range(len(self.stack)-1, -1, -1):
            if self.stack[n][0] == tag:
                del self.stack[n:]
                break
        if tag in {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'aside', 'li', 'tr', 'td', 'th'}:
            self.parts.append('\n')

    def handle_data(self, value):
        if not any(v for _, v in self.stack):
            self.parts.append(value)


def reading_text(page):
    parser = _ReadingText()
    parser.feed('\n'.join(page.get('body', []) + page.get('asides', [])))
    return ''.join(parser.parts)


def sweep_pages(pages, records, expected_pages, *, lexicon=None):
    """Check every physical page's actual reading text before page-turn joins."""
    expected = sorted(expected_pages)
    actual = [p['pno'] for p in pages]
    if sorted(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError('Text quality sweep does not cover every selected source page exactly once')
    result = {}
    for page in pages:
        pno = page['pno']
        metric = inspect(reading_text(page), lexicon=lexicon)
        resolution = records[pno]['source']
        metric['resolution'] = resolution
        # A complete source-page disclosure has no purported reading text.
        metric['passed'] = not metric['needs_read']
        result[pno] = metric
    return dict(version=VERSION, expected_pages=expected, checked_pages=sorted(result),
        pages=result, passed=all(m['passed'] for m in result.values()),
        thresholds=dict(min_page_words=MIN_PAGE_WORDS, dictionary_miss_rate=MISS_RATE,
            min_case_words=MIN_CASE_WORDS, case_flip_rate=CASE_FLIP_RATE,
            case_dictionary_miss_rate=CASE_MISS_RATE, severe_case_transitions=SEVERE_CASE_TRANSITIONS, window_words=WINDOW_WORDS,
            window_dictionary_miss_rate=WINDOW_MISS_RATE))


def validate_sweep(sweep, expected_pages):
    """The packaging gate cannot accept an incomplete or rejected quality sweep."""
    expected = sorted(expected_pages)
    pages = sweep.get('pages', {})
    observed = sorted(int(p) for p in pages)
    if (sweep.get('version') != VERSION or sweep.get('expected_pages') != expected
            or sweep.get('checked_pages') != expected or observed != expected
            or not sweep.get('passed') or any(not m.get('passed') or m.get('needs_read') for m in pages.values())):
        raise ValueError('Every source page requires a passing text quality sweep or retained source disclosure')

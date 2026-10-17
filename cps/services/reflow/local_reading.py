"""Bounded, explicitly unverified OCR readings tied to original source pixels.

Engine scores are not probabilities. Every locally transcribed word remains
uncertain, including words that agree with an existing OCR text layer. Paragraph
identities and word rectangles allow independent source disclosure on rejection.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

from . import ocr, vision_reading

VERSION = 'local-reading-1'
MAX_PIXELS = 8_000_000
TIMEOUT = 120


def recognize(display, cache):
    """Recognize at 2x, capped at eight million pixels and 120 seconds per tile."""
    executable, version, data = ocr._engine('eng')
    pix = display.pixmap(scale=2, max_pixels=MAX_PIXELS)
    image = pix.tobytes('png')
    identity = dict(adapter=VERSION, image=hashlib.sha256(image).hexdigest(),
                    engine=version, data=data, rect=list(display.rect))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    target = root / (key + '.json')
    if target.exists() and target.stat().st_size <= ocr.MAX_OUTPUT_BYTES:
        try:
            saved = json.loads(target.read_text())
            if saved['identity'] == identity and saved['sha256'] == hashlib.sha256(
                    json.dumps(saved['words'], sort_keys=True).encode()).hexdigest():
                _validate_words(saved['words'], display.rect)
                return saved['words'], dict(identity, cached=True)
        except (ValueError, KeyError, TypeError):
            pass
    tiles = [(image, 0, pix.width)]
    if pix.width >= pix.height * 1.2:
        from PIL import Image
        picture = Image.open(io.BytesIO(image))
        middle = pix.width // 2
        tiles = []
        for left, right in ((0, middle), (middle, pix.width)):
            buffer = io.BytesIO()
            picture.crop((left, 0, right, pix.height)).save(buffer, format='PNG')
            tiles.append((buffer.getvalue(), left, right-left))
    output = []
    sx, sy = display.rect.width / pix.width, display.rect.height / pix.height
    for tile_index, (encoded, offset, width) in enumerate(tiles):
        with tempfile.TemporaryDirectory(prefix='local-ocr-', dir=root) as temporary:
            directory = Path(temporary)
            source, table = directory / 'page.png', directory / 'result.tsv'
            source.write_bytes(encoded)
            status, _, _ = ocr._invoke([executable, str(source), str(directory / 'result'),
                '-l', 'eng', '--psm', '3', 'tsv'], directory, timeout=TIMEOUT,
                should_stop=None, watched=(table,))
            if status or not table.exists():
                raise ocr.OCRFailed('Local page recognition failed.')
            for row in csv.DictReader(io.StringIO(table.read_text()), delimiter='\t', quoting=csv.QUOTE_NONE):
                if row['level'] != '5' or not row['text'].strip():
                    continue
                x, y, w, h = [int(row[name]) for name in ('left', 'top', 'width', 'height')]
                if not (0 <= x < x+w <= width and 0 <= y < y+h <= pix.height):
                    raise ocr.OCRFailed('Local recognition returned invalid source geometry.')
                output.append(dict(text=row['text'], bbox=[(x+offset)*sx, y*sy,
                    (x+offset+w)*sx, (y+h)*sy], confidence=float(row['conf']),
                    paragraph=[tile_index, int(row['block_num']), int(row['par_num'])],
                    line=int(row['line_num'])))
                if len(output) > ocr.MAX_WORDS:
                    raise ocr.OCRFailed('Local recognition exceeded its word limit.')
    _validate_words(output, display.rect)
    saved = dict(identity=identity, words=output,
        sha256=hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest())
    encoded = json.dumps(saved, ensure_ascii=False)
    if len(encoded.encode()) > ocr.MAX_OUTPUT_BYTES:
        raise ocr.OCRFailed('Local recognition exceeded its receipt limit.')
    with tempfile.NamedTemporaryFile(mode='w', dir=root, prefix='.receipt-', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return output, dict(identity, cached=False)


def _validate_words(words, rect):
    if not isinstance(words, list) or len(words) > ocr.MAX_WORDS:
        raise ValueError('Invalid local recognition word inventory')
    for word in words:
        if not isinstance(word['text'], str) or not word['text'].strip():
            raise ValueError('Invalid local recognition text')
        box = word['bbox']
        if (len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= rect.width
                or not 0 <= box[1] < box[3] <= rect.height
                or not math.isfinite(word['confidence']) or not 0 <= word['confidence'] <= 100
                or len(word['paragraph']) != 3
                or any(type(v) is not int for v in word['paragraph']) or type(word['line']) is not int):
            raise ValueError('Invalid local recognition evidence')


def read(display, cache, reason):
    words, receipt = recognize(display, cache)
    if not words:
        raise ocr.OCRFailed('Local recognition found no readable words.')
    paragraphs = {}
    for word in words:
        paragraphs.setdefault(tuple(word['paragraph']), []).append(word)
    blocks, units = [], []
    for index, owned in enumerate(paragraphs.values()):
        ident = 'ocr' + str(index)
        box = vision_reading.union([word['bbox'] for word in owned])
        text = ' '.join(word['text'] for word in owned)
        blocks.append(dict(id=ident, role='body', text=text, marker='',
            bbox=[box[0]/display.rect.width*1000, box[1]/display.rect.height*1000,
                  box[2]/display.rect.width*1000, box[3]/display.rect.height*1000],
            low_confidence=list(range(len(text.split()))),
            continues_previous=False, continues_next=False))
        units.append(dict(id=ident, words=owned, mean_confidence=sum(w['confidence'] for w in owned)/len(owned),
                          low_confidence_tokens=[i for i,w in enumerate(owned) if w['confidence'] < 85]))
    return dict(answer=vision_reading.validate({'blocks': blocks}), local_ocr=True,
        local_ocr_evidence=receipt, local_ocr_units=units, local_ocr_reason=reason,
        cached=receipt['cached'], new_cost_usd=0., unverified_source_reading=True)

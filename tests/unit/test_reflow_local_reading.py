"""Synthetic raster readings retain their own source occurrence and uncertainty."""
import shutil

import pymupdf
import pytest

from cps.services.reflow import unit_recovery, vision_pipeline, vision_reading

pytestmark = pytest.mark.unit


def test_unavailable_reading_uses_local_words_without_releasing_a_financial_hold(tmp_path):
    from cps.services.reflow.ledger import Ledger
    ledger = Ledger(tmp_path/'ledger.jsonl', 1)
    ledger.reserve_attempt('synthetic-request', .25)
    class Reader:
        def read_unit(self, image, **kwargs):
            return dict(degraded_source=True, failure='Reading unavailable',
                        recovery_events=[dict(transition='UNKNOWN_FULL_BOUND_RETAINED')])
    controller = unit_recovery.Controller(Reader(), [0], tmp_path)
    answer = {'blocks': [dict(id='o1', role='body', text='A small bridge crosses the stream.',
        marker='', bbox=[10, 10, 900, 100], low_confidence=list(range(7)),
        continues_previous=False, continues_next=False)]}
    called = []
    def local(reason):
        called.append(reason)
        return dict(answer=answer, local_ocr=True, cached=False, new_cost_usd=0.)
    reading = controller.read(0, b'synthetic-raster', local_read=local)
    assert reading['answer'] == answer and reading['local_ocr'] is True
    assert called == ['Reading unavailable']
    assert controller.events == [dict(transition='UNKNOWN_FULL_BOUND_RETAINED')]
    assert Ledger(tmp_path/'ledger.jsonl', 1).pending_usd() == .25
    assert controller.read(0, b'synthetic-raster', local_read=lambda reason: pytest.fail('repeated OCR')) == reading


def test_local_ocr_agreement_with_a_layer_never_certifies_the_words(tmp_path):
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    doc = pymupdf.open()
    doc.new_page(width=300, height=400)
    text = 'The bridge crosses the small stream.'
    boxes = [dict(text=token, bbox=[20+i*35, 30, 53+i*35, 42])
             for i, token in enumerate(text.split())]
    answer = {'blocks': [block(text, bbox=(50, 50, 950, 200), low=range(6))]}
    class Package:
        def image(self, *args): pass
    page, report = vision_pipeline.compile_page(answer, boxes, SourceDisplay(doc, 0),
        0, Package(), {}, pixel_words=boxes, local_ocr=True)
    body = ''.join(page['body'])
    assert text.replace(' ', '') in ''.join(__import__('re').sub('<[^>]+>', '', body).split())
    assert 'class="uncertain"' in body
    assert report['marks'] == 1
    assert 'Unverified local OCR reading' in ''.join(page['asides'])
    assert 'Low-confidence model reading' not in ''.join(page['asides'])
    doc.close()


@pytest.mark.skipif(not shutil.which('tesseract'), reason='needs installed OCR engine')
def test_empty_cache_no_key_raster_has_live_source_linked_unverified_words(tmp_path):
    from cps.services.reflow.ledger import Ledger
    from tests.unit.test_reflow_vision_pilot import documents, assert_links_resolve
    source = pymupdf.open()
    page = source.new_page(width=420, height=500)
    lines = ['The narrow bridge crosses the quiet stream.',
             'The green garden contains a small wooden bench.',
             'A visitor walks along the path before sunset.']
    for index, line in enumerate(lines):
        page.insert_text((30, 70+index*35), line, fontsize=13)
    image = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes('png')
    doc = pymupdf.open()
    doc.new_page(width=420, height=500).insert_image(pymupdf.Rect(0, 0, 420, 500), stream=image)
    cache = tmp_path / 'readings'
    reader = vision_reading.Reader(None, cache, Ledger(tmp_path/'ledger.jsonl', 0))
    path = tmp_path / 'local.epub'
    result = vision_pipeline.convert(doc, path, reader)
    docs = documents(path)
    assert_links_resolve(docs)
    live = ''.join(''.join(element.itertext()) for name, root in docs.items()
        if not name.endswith('/vision-about.xhtml')
        for element in root.iter() if element.get('data-source-block'))
    for line in lines:
        assert line in live
    assert result['pages'][0]['source'] == 'vision'
    assert result['pages'][0]['local_ocr'] is True
    assert result['pages'][0]['unverified_source_reading'] is True
    assert result['pages'][0]['marks'] >= 1
    assert result['pages'][0]['new_cost_usd'] == 0
    assert list(cache.glob('*.json')) == [], 'OCR receipts must not impersonate paid readings'
    doc.close()
    source.close()


def test_local_recognition_uses_bounded_owned_scratch_and_retains_occurrence_scores(tmp_path, monkeypatch):
    from cps.services.reflow import local_reading, ocr
    from cps.services.reflow.source_display import SourceDisplay
    doc = pymupdf.open()
    doc.new_page(width=300, height=400)
    monkeypatch.setattr(ocr, '_engine', lambda language: ('synthetic-engine', 'synthetic-version', 'synthetic-data'))
    calls = []
    def invoke(args, directory, **kwargs):
        calls.append((args, kwargs))
        assert directory.parent == tmp_path
        assert kwargs == dict(timeout=120, should_stop=None, watched=(directory/'result.tsv',))
        assert (directory/'page.png').exists()
        (directory/'result.tsv').write_text('level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
            '5\t1\t1\t1\t1\t1\t20\t40\t50\t20\t96\tbridge\n'
            '5\t1\t1\t1\t1\t2\t90\t40\t50\t20\t61\tbridge\n')
        return 0, '', ''
    monkeypatch.setattr(ocr, '_invoke', invoke)
    record = local_reading.read(SourceDisplay(doc, 0), tmp_path, 'No reading available')
    assert record['answer']['blocks'][0]['text'] == 'bridge bridge'
    assert record['answer']['blocks'][0]['low_confidence'] == [0, 1]
    owned = record['local_ocr_units'][0]
    assert owned['low_confidence_tokens'] == [1]
    assert owned['words'][0]['bbox'] != owned['words'][1]['bbox']
    assert [w['confidence'] for w in owned['words']] == [96, 61]
    again = local_reading.read(SourceDisplay(doc, 0), tmp_path, 'No reading available')
    assert again['local_ocr_evidence']['cached'] and len(calls) == 1
    assert not list(tmp_path.glob('local-ocr-*')), 'owned invocation scratch must be removed'
    doc.close()

"""Synthetic repeated tokens bind confidence only to a unique geometric owner."""
import json
from xml.etree import ElementTree as ET

import pytest

from cps.services.reflow import annotate, assemble, enriched_source

pytestmark = pytest.mark.unit


def paragraph(text, box):
    return assemble.Element('p', runs=[['t', text]], pno=0, bbox=box,
                            line_boxes=[box])


def prepared(elements, box, *, angle=0, frame=(0, 0, 200, 300)):
    book = assemble.Book(elements=elements, pages={0: elements})
    record = {'token': 'Mark', 'candidates': ['Mask'], 'source_bbox': box}
    provenance = {'layer': 'ocr', 'orientation': angle, 'page_rect': frame,
                  'uncertain_words': 1, 'uncertain': [record]}
    return enriched_source.prepare_source_page(book, 0, provenance, [record])


def marks(source):
    tree = ET.fromstring('<root>' + source.html + '</root>')
    return [[node.text for node in p.iter('span')
             if node.get('class') == annotate.CLASS] for p in tree.findall('p')]


def test_repeated_word_marks_only_its_own_second_paragraph():
    source = prepared([paragraph('Mark is clear.', (10, 10, 160, 30)),
                       paragraph('Mark is damaged.', (10, 80, 160, 100))],
                      [10, 80, 40, 100])
    assert marks(source) == [[], ['Mark']]
    report = source.report()
    assert report['marked'] == 1 and report['placed_record_indices'] == [0]
    assert report['unplaced_record_indices'] == []
    assert report['uncertain'][0]['source_bbox'] == [10, 80, 40, 100]


@pytest.mark.parametrize('kind', ['same-block', 'overlap', 'outside', 'invalid'])
def test_ambiguous_or_unowned_geometry_stays_disclosed_unplaced(kind):
    elements = [paragraph('Mark is clear.', (10, 10, 160, 30)),
                paragraph('Mark is damaged.', (10, 80, 160, 100))]
    box = [10, 80, 40, 100]
    if kind == 'same-block':
        elements = [paragraph('Mark is clear. Mark is damaged.', (10, 10, 160, 100))]
    elif kind == 'overlap':
        elements[0].bbox = elements[1].bbox
        elements[0].line_boxes = elements[1].line_boxes
    elif kind == 'outside':
        box = [10, 180, 40, 200]
    else:
        box = [10, 80, float('nan'), 100]
    source = prepared(elements, box)
    assert not any(marks(source))
    report = source.report()
    assert report['marked'] == 0 and report['placed_record_indices'] == []
    assert report['unplaced_record_indices'] == [0]
    assert report['uncertain'][0]['token'] == 'Mark'


def test_geometry_without_source_context_cannot_guess_the_first_match():
    html = '<p>Mark is clear. Mark is damaged.</p>'
    record = {'token': 'Mark', 'candidates': [], 'source_bbox': [10, 80, 40, 100]}
    out, unresolved, marked, positions = annotate.annotate_page(
        html, [record], with_placements=True)
    assert out == html and unresolved == [record]
    assert marked == 0 and positions == []


def test_rotated_original_geometry_selects_the_correct_reading_block():
    # A 90-degree turn maps the second source box to reading x=200..220, y=10..40.
    source = prepared([paragraph('Mark is clear.', (270, 10, 290, 160)),
                       paragraph('Mark is damaged.', (200, 10, 220, 160))],
                      [10, 80, 40, 100], angle=90)
    assert marks(source) == [[], ['Mark']]
    assert source.report()['placed_record_indices'] == [0]


def test_an_existing_mark_cannot_make_a_repeated_token_look_unique():
    html = '<p><span class="reflow-uncertain">Mark</span> is clear. Mark is damaged.</p>'
    record = {'token': 'Mark', 'source_bbox': [10, 10, 40, 30]}
    out, placed = annotate.mark_uncertain(html, [record], source_ranges={0: (0, len(html))})
    assert out == html and placed == []


def test_unplaced_geometry_is_visible_in_the_original_page_document():
    from cps.services.reflow import build_epub
    source = prepared([paragraph('Mark is clear. Mark is damaged.', (10, 10, 160, 100))],
                      [10, 80, 40, 100])
    document = build_epub._original_document(
        {'page': 0, 'full': 'images/synthetic.png', 'details': [],
         'source_uncertainty': source.report()}, 'chapter.xhtml', 'en')
    tree = ET.fromstring(document)
    listed = [''.join(node.itertext()) for node in tree.iter('{http://www.w3.org/1999/xhtml}li')]
    assert listed == ['Mark (not highlighted)']
    assert source.report()['unplaced_record_indices'] == [0]

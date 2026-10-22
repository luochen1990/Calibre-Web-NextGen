"""Synthetic printed rows retain one owner across different recognition boxes."""
import pymupdf
import pytest

from cps.services.reflow import vision_pipeline
from cps.services.reflow.source_display import SourceDisplay
from tests.unit.test_reflow_vision_pilot import answer, block

pytestmark = pytest.mark.unit


class Package:
    def disclosure_image(self, *args):
        pass


@pytest.mark.parametrize('leaders',[False,True])
def test_native_row_is_not_repeated_through_neighbor_optical_territory(tmp_path,leaders):
    with pymupdf.open() as doc:
        page = doc.new_page(width=500, height=250)
        first='Entry for the northern road '+('. '*30 if leaders else '')+'12'
        second='Entry for the western road '+('. '*30 if leaders else '')+'15'
        page.insert_text((30, 60), first, fontsize=12)
        page.insert_text((30, 90), second, fontsize=12)
        words = [dict(text=w[4], bbox=list(w[:4]), confidence=99) for w in page.get_text('words')]
        optical = [dict(w, bbox=[x + .4 for x in w['bbox']]) for w in words]
        row_length=len(words)//2
        regions = {'b1': [[25, 40, 490, 100]], 'b2': [[25, 75, 490, 100]],
                   '_instances': {vision_pipeline._source_instance(w): 'b1' if i < row_length else 'b2'
                                  for i, w in enumerate(optical)}}
        reading = answer(dict(block(first), id='b1'),dict(block(second), id='b2'))
        _, record = vision_pipeline.compile_page(
            reading, words, SourceDisplay(doc, 0), 0, Package(), {}, pixel_words=optical,
            regions=regions, body_owners={},
            note_context={'cache': tmp_path/'cache', 'labels': {}, 'catalogue': {}, 'refs': {}})
    if not leaders:
        assert [b['published_tokens']for b in record['blocks']]==[first.split(),second.split()]
    lexical=lambda tokens:[t for t in tokens if any(c.isalnum()for c in t)]
    assert [lexical(b['published_tokens']) for b in record['blocks']] == [
        'Entry for the northern road 12'.split(), 'Entry for the western road 15'.split()]


def test_two_distinct_printed_occurrences_keep_both_readings(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=500, height=250)
        text = 'The repeated line remains clearly printed.'
        page.insert_text((30, 60), text, fontsize=12)
        page.insert_text((30, 100), text, fontsize=12)
        words = [dict(text=w[4], bbox=list(w[:4]), confidence=99) for w in page.get_text('words')]
        regions = {'b1': [[25, 40, 350, 65]], 'b2': [[25, 80, 350, 105]]}
        reading = answer(block(text, ident='b1'), block(text, ident='b2'))
        _, record = vision_pipeline.compile_page(
            reading, words, SourceDisplay(doc, 0), 0, Package(), {}, pixel_words=words,
            regions=regions, body_owners={},
            note_context={'cache': tmp_path/'cache', 'labels': {}, 'catalogue': {}, 'refs': {}})
    assert [b['published_tokens'] for b in record['blocks']] == [text.split(), text.split()]


def test_complete_short_index_row_owns_its_occurrences_over_a_borrowed_tail(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((30, 60), 'Birch 2nd 66', fontsize=12)
        page.insert_text((30, 85), 'Birch 12th 92', fontsize=12)
        words = [dict(text=w[4], bbox=list(w[:4]), confidence=99) for w in page.get_text('words')]
        regions = {'b1': [[25, 40, 200, 90]], 'b2': [[25, 70, 200, 90]]}
        reading = answer(block('Birch 2nd 66 Birch 12th 92', ident='b1'),
                         block('Birch 12th 92', ident='b2'))
        _, record = vision_pipeline.compile_page(reading, words, SourceDisplay(doc, 0),
            0, Package(), {}, pixel_words=words, regions=regions, body_owners={},
            note_context={'cache': tmp_path/'cache', 'labels': {}, 'catalogue': {}, 'refs': {}})
    assert [b['published_tokens'] for b in record['blocks']] == [
        ['Birch', '2nd', '66'], ['Birch', '12th', '92']]


@pytest.mark.parametrize('tab_gap', [False, True])
@pytest.mark.parametrize('clipped_tail', [False, True])
def test_short_index_row_ownership_does_not_require_guessing_its_damaged_reference(tmp_path, tab_gap, clipped_tail):
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((30, 60), 'Birch 2nd 66', fontsize=12)
        if tab_gap:
            for x, text in ((30, 'Alder'), (170, '12th'), (215, 'X'), (245, '92')):
                page.insert_text((x, 85), text, fontsize=12)
        else:
            page.insert_text((30, 85), 'Alder 12th X 92', fontsize=12)
        words = [dict(text=w[4], bbox=list(w[:4]), confidence=99) for w in page.get_text('words')]
        regions = {'b1': [[25, 40, 280, 90]], 'b2': [[25, 70, 230 if clipped_tail and tab_gap else 280, 90]]}
        reading = answer(block('Birch 2nd 66 Alder 12th X 92', ident='b1'),
                         block('Alder 12th Y 92', ident='b2'))
        _, record = vision_pipeline.compile_page(reading, words, SourceDisplay(doc, 0),
            0, Package(), {}, pixel_words=words, regions=regions, body_owners={},
            note_context={'cache': tmp_path/'cache', 'labels': {}, 'catalogue': {}, 'refs': {}})
    assert [b['published_tokens'] for b in record['blocks']] == [
        ['Birch', '2nd', '66'], ['Alder', '12th', 'X', '92']]


def test_index_row_tail_cannot_cross_a_new_lexical_column_label():
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        for x, text in ((30, 'Birch'), (100, '2nd'), (135, '66'), (165, 'Other'), (205, '92')):
            page.insert_text((x, 60), text, fontsize=12)
        words = [dict(text=w[4], bbox=list(w[:4])) for w in page.get_text('words')]
        regions = {'b1': [[25, 40, 180, 65]]}
        reading = answer(block('Birch 2nd 92', ident='b1'))
        vision_pipeline._reserve_layer_instances(reading, words, SourceDisplay(doc, 0), regions)
        foreign = next(w for w in words if w['text']=='92')
        assert regions.get('_instances', {}).get(vision_pipeline._source_instance(foreign)) != 'b1'


@pytest.mark.parametrize('shift', [0, .4])
def test_owned_caption_survives_a_differently_split_native_inventory(tmp_path, shift):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=220)
        page.insert_text((30, 50), 'The garden remains clearly visible.', fontsize=12)
        page.insert_text((160, 80), 'The view from the path.', fontsize=12)
        page.insert_text((30, 110), 'Its trees remain green throughout summer.', fontsize=12)
        optical = [dict(text=w[4], bbox=list(w[:4]), confidence=99)
                   for w in page.get_text('words')]
        native = []
        for word in optical:
            box = word['bbox']
            if word['text'] == 'from':
                cut = box[0] + .72 * (box[2] - box[0])
                native.extend([dict(word, text='fro', bbox=[box[0], box[1], cut, box[3]]),
                               dict(word, text='m', bbox=[cut, box[1], box[2], box[3]])])
            else:
                native.append(dict(word, bbox=[v + shift for v in box]))
        caption = [w for w in optical if 60 < w['bbox'][1] < 90]
        regions = {'b1': [[25, 30, 380, 115]], 'c1': [[155, 60, 300, 85]],
                   '_instances': {vision_pipeline._source_instance(w): 'c1' for w in caption}}
        reading = answer(block('The garden remains clearly visible. Its trees remain green throughout summer.', ident='b1'),
                         dict(block('The view from the path.', ident='c1'), role='caption'))
        package = Package()
        package.image = lambda *args: None
        _, record = vision_pipeline.compile_page(reading, native, SourceDisplay(doc, 0),
            0, package, {}, pixel_words=optical, regions=regions, body_owners={},
            note_context={'cache': tmp_path/'cache', 'labels': {}, 'catalogue': {}, 'refs': {}})
    body = next(b for b in record['blocks'] if b['id'] == 'b1')
    assert body['published_tokens'] == reading['blocks'][0]['text'].split()


def test_conflicting_optical_owners_cannot_rebind_a_native_occurrence():
    with pymupdf.open() as doc:
        doc.new_page(width=200, height=150)
        word = dict(text='garden', bbox=[30, 40, 65, 52])
        regions = {'_instances': {('garden', (29.8, 40, 65, 52)): 'b1',
                                  ('garden', (30.2, 40, 65, 52)): 'b2'}}
        reading = answer(block('garden', ident='b1'), block('garden', ident='b2'))
        vision_pipeline._reserve_layer_instances(reading, [word], SourceDisplay(doc, 0), regions)
    assert vision_pipeline._source_instance(word) not in regions['_instances']


def test_broad_body_territory_does_not_adopt_unclaimed_art_lettering():
    with pymupdf.open() as doc:
        doc.new_page(width=400, height=220)
        prose = dict(text='garden', bbox=[165, 65, 200, 77])
        label = dict(text='DRAWING', bbox=[220, 65, 275, 77])
        reading = answer(block('garden', ident='b1'),
                         dict(block('An original drawing.', ident='f1'), role='figure'))
        regions = {'b1': [[25, 30, 380, 115]], 'f1': [[160, 60, 300, 85]],
                   '_instances': {vision_pipeline._source_instance(prose): 'b1'}}
        result = vision_pipeline._owned_block_words(reading['blocks'][0], reading,
            [prose, label], SourceDisplay(doc, 0), regions)
    assert result == [prose]


def test_numbered_short_rows_own_their_omitted_optical_labels(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=160)
        page.insert_text((30, 60), '1. RIVER, water', fontsize=12)
        page.insert_text((30, 80), '2. GARDEN, flowers', fontsize=12)
        native = [dict(text=w[4], bbox=list(w[:4]), confidence=99)
                  for w in page.get_text('words')]
        optical = [w for w in native if not w['text'][0].isdigit()]
        reading = answer(block('1. RIVER, water', ident='b1'),
                         block('2. GARDEN, flowers', ident='b2'))
        regions = {'b1': [[25, 40, 250, 63]], 'b2': [[25, 50, 250, 83]]}
        display = SourceDisplay(doc, 0)
        vision_pipeline._reserve_layer_instances(reading, optical, display, regions)
        vision_pipeline._reserve_layer_instances(reading, native, display, regions)
        labels = [w for w in native if w['text'][0].isdigit()]
        assert [regions['_instances'].get(vision_pipeline._source_instance(w))
                for w in labels] == ['b1', 'b2']
        _, record = vision_pipeline.compile_page(reading, native, display, 0,
            Package(), {}, pixel_words=optical, regions=regions, body_owners={},
            note_context={'cache': tmp_path/'cache', 'labels': {}, 'catalogue': {}, 'refs': {}})
    assert [b['published_tokens'] for b in record['blocks']] == [
        ['1.', 'RIVER,', 'water'], ['2.', 'GARDEN,', 'flowers']]

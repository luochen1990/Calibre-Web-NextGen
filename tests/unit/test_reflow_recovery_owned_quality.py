"""Synthetic unit failures retain independently readable neighbours."""
import copy
from types import SimpleNamespace

import pytest

from cps.services.reflow import unit_recovery as ur, vision_pipeline as vp
from tests.unit import test_round4_defects as ink

pytestmark = pytest.mark.unit


def test_recovered_omission_keeps_its_geometry_owner_and_safe_sibling(tmp_path, monkeypatch):
    page = ink.InkPage()
    page.text('The clear sample remains readable.', 'healthy', y=45)
    omitted = page.text(['The second printed sample remains clear',
                         'while the soft light crosses the panel.'], 'omitted', y=145, size=8)
    page.blocks.remove(omitted)
    align = vp.vision_reading.align
    def rejected_recovery(text, *args, **kwargs):
        result = align(text, *args, **kwargs)
        if 'second printed sample' in text:
            result.update(tokens=['zzqv', 'rrjx', 'ppqx'], marked={0, 1, 2}, fallback=True)
        return result
    monkeypatch.setattr(vp.vision_reading, 'align', rejected_recovery)
    _, report, docs = ink.convert_ink(tmp_path, monkeypatch, [page])
    assert report['pages'][0]['source'] == 'vision'
    assert 'The clear sample remains readable.' in ' '.join(ink.plain(root) for root in docs.values())
    units = report['pages'][0]['degraded_units']
    assert len(units) == 1 and units[0]['block'].startswith('recovered-')
    assert units[0]['source_image'] and units[0]['source_reference']
    assert not any(word in ' '.join(ink.plain(root) for root in docs.values()) for word in ['zzqv', 'rrjx', 'ppqx'])


def test_whole_page_supersedes_stale_unit_without_erasing_its_history(tmp_path):
    controller = ur.Controller(SimpleNamespace(key=None), [0], tmp_path)
    controller.recover(ur.UnitFailure([dict(page=0, block='failed')], 'synthetic_unit', ValueError('First unit failed.')))
    controller.recover(ur.UnitFailure([dict(page=0, block=None)], 'synthetic_page', ValueError('Page ownership is unproved.')))
    assert set(controller.degraded) == {(0, None)}
    assert controller.events[0]['units'] == [dict(page=0, block='failed')]
    assert controller.events[-1]['source_page_escalations'][0]['previous_units'][0]['block'] == 'failed'


def test_initial_quality_failure_discloses_only_the_bad_unit(tmp_path, monkeypatch):
    page = ink.InkPage()
    page.text('The clear sample remains readable.', 'healthy', y=45)
    page.text(['qVrTx mNpRs wXyZq aBcDe fGhIj kLmNo',
               'pQrSt uVwXy zAbCd eFgHi jKlMn oPqRs'], 'damaged', y=145, size=7)
    _, report, docs = ink.convert_ink(tmp_path, monkeypatch, [page])
    assert report['pages'][0]['source'] == 'vision'
    assert 'The clear sample remains readable.' in ' '.join(ink.plain(root) for root in docs.values())
    units = report['pages'][0]['degraded_units']
    assert {unit['block'] for unit in units} == {'damaged'}
    assert units[0]['source_image'] and units[0]['source_reference']


def test_joined_quality_failure_retains_only_its_fragment_owner(tmp_path):
    controller = ur.Controller(SimpleNamespace(key=None), [0], tmp_path)
    healthy = '<p><span data-source-page="0" data-source-block="healthy">The clear sample remains readable.</span></p>'
    damaged = '<p><span data-source-page="0" data-source-block="damaged">qVrTx mNpRs wXyZq aBcDe fGhIj kLmNo pQrSt uVwXy zAbCd eFgHi jKlMn oPqRs</span></p>'
    page = dict(pno=0, anchor='<span id="pg_0000"></span>', body=[healthy, damaged], asides=[])
    record = dict(source='vision', blocks=[dict(id='healthy', role='body'), dict(id='damaged', role='body')])
    controller.records = {0: record}
    controller.register_page(page, record)
    with ur.activated(controller), pytest.raises(ur.UnitFailure) as failure:
        vp._compiled_text_quality([page], controller.records, [0])
    assert failure.value.units == [dict(page=0, block='damaged')]


def test_post_join_image_alt_is_gated_at_its_source_unit(tmp_path):
    controller = ur.Controller(SimpleNamespace(key=None), [0], tmp_path)
    healthy = '<p><span data-source-page="0" data-source-block="healthy">The clear sample remains readable.</span></p>'
    damaged = '<figure id="art"><img src="images/sample.png" alt="zzqv rrjx ppqx"/></figure>'
    page = dict(pno=0, anchor='<span id="pg_0000"></span>', body=[healthy, damaged], asides=[])
    record = dict(source='vision', blocks=[dict(id='healthy', role='body'), dict(id='art', role='figure')])
    controller.records = {0: record}
    controller.register_page(page, record)
    with ur.activated(controller), pytest.raises(ur.UnitFailure) as failure:
        vp._compiled_text_quality([page], controller.records, [0])
    assert failure.value.units == [dict(page=0, block='art')]


def test_hidden_script_does_not_discard_the_visible_latin_reading(tmp_path, monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pixels
    from tests.unit.test_reflow_text_quality import SyntheticReader
    from tests.unit.test_reflow_vision_pilot import answer, block
    text = 'The clear sample remains readable.'
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        page.insert_text((30, 60), text)
        page.insert_font(fontname='alphabet', fontbuffer=pymupdf.Font('helv').buffer)
        page.insert_text((30, 100), 'αλφαβήτα γαμμαδέλτα', fontname='alphabet', render_mode=3)
        assert 'αλφα' in page.get_text('text')
        optical = [dict(text=w[4], bbox=list(w[:4])) for w in page.get_text('words') if w[1] < 70]
        monkeypatch.setattr(vp, 'clean_native', lambda *args: False)
        monkeypatch.setattr(vision_pixels, 'words', lambda *args: copy.deepcopy(optical))
        reader = SyntheticReader(answer(block(text, bbox=(0, 0, 1000, 1000))))
        report = vp.convert(doc, tmp_path/'visible.epub', reader)
    assert report['pages'][0]['source'] == 'vision'
    assert not any(unit.get('block') is None for unit in report.get('degraded_units', []))
    assert report['pages'][0]['words'] == len(text.split())


def test_joined_note_rejection_keeps_clean_original_and_body_units(tmp_path):
    controller = ur.Controller(SimpleNamespace(key=None), [0, 1], tmp_path)
    aside = ('<aside id="v4-note-0-noteA" class="footnote"><p>'
             '<span data-source-page="0" data-source-block="noteA">The clear sample remains readable.</span></p><p>'
             '<span data-source-page="1" data-source-block="noteB">qVrTx mNpRs wXyZq aBcDe fGhIj kLmNo pQrSt uVwXy zAbCd eFgHi jKlMn oPqRs</span></p></aside>')
    pages = [dict(pno=0, anchor='<span id="pg_0000"></span>', body=[], asides=[aside]),
             dict(pno=1, anchor='<span id="pg_0001"></span>', body=['<p><span data-source-page="1" data-source-block="healthy">The second clear sample remains readable.</span></p>'], asides=[])]
    records = {0: dict(source='vision', blocks=[dict(id='noteA', role='footnote')]),
               1: dict(source='vision', blocks=[dict(id='noteB', role='footnote'), dict(id='healthy', role='body')])}
    controller.records = records
    for page in pages:
        controller.register_page(page, records[page['pno']])
    with ur.activated(controller), pytest.raises(ur.UnitFailure) as failure:
        vp._compiled_text_quality(pages, records, [0, 1])
    assert failure.value.units == [dict(page=1, block='noteB')]


def test_rejected_graphic_wording_cannot_demand_a_removed_note_owner(tmp_path):
    from cps.services.reflow import vision_reading
    controller = ur.Controller(SimpleNamespace(key=None), [0], tmp_path)
    template = dict(marker='', low_confidence=[], continues_previous=False, continues_next=False, bbox=[0, 0, 1000, 1000])
    raw = dict(blocks=[dict(template, id='graphic', role='table', text='A clear row <n:note>'),
                       dict(template, id='note', role='footnote', text='A short printed note.'),
                       dict(template, id='healthy', role='body', text='The clear sample remains readable.')])
    controller.answers[0] = copy.deepcopy(raw)
    controller.degraded = {(0, ident): dict(page=0, block=ident) for ident in ['graphic', 'note']}
    selected = controller.filter_answer(0, raw)
    vision_reading.validate(selected)
    assert next(block['text'] for block in selected['blocks'] if block['id'] == 'healthy') == raw['blocks'][2]['text']
    assert next(block['text'] for block in selected['blocks'] if block['id'] == 'graphic') == 'Original source image'
    assert controller.answers[0] == raw


def test_bad_joined_paragraph_tail_does_not_discard_its_clean_source_page(tmp_path):
    controller = ur.Controller(SimpleNamespace(key=None), [0, 1], tmp_path)
    clean = 'The clear sample remains readable. ' * 5
    damaged = 'qVrTx mNpRs wXyZq aBcDe fGhIj kLmNo pQrSt uVwXy zAbCd eFgHi jKlMn oPqRs'
    fragment = ('<p><span data-source-page="0" data-source-block="good">'+clean+'</span>'
                '<span id="pg_0001"></span><span data-source-page="1" data-source-block="bad">'+damaged+'</span></p>')
    pages = [dict(pno=0, anchor='<span id="pg_0000"></span>', body=[fragment], asides=[]),
             dict(pno=1, anchor='', body=[], asides=[])]
    records = {0: dict(source='vision', blocks=[dict(id='good', role='body')]),
               1: dict(source='vision', blocks=[dict(id='bad', role='body')])}
    controller.records = records
    for page in pages:
        controller.register_page(page, records[page['pno']])
    with ur.activated(controller), pytest.raises(ur.UnitFailure) as failure:
        vp._compiled_text_quality(pages, records, [0, 1])
    assert failure.value.units == [dict(page=1, block=None)]


def test_rejected_table_retains_its_admitted_geometry_and_safe_heading(tmp_path):
    page = ink.InkPage()
    page.text('A clear sample heading', 'heading', role='heading', y=45)
    graphic = page.text('A clear sample row', 'graphic', role='table', y=145)
    graphic['bbox'] = [-5, -5, -1, -1]
    _, words, raw = page.finish()
    display = SimpleNamespace(rect=ink.pymupdf.Rect(0, 0, ink.W, ink.H))
    admitted, _ = vp.bind_geometry(copy.deepcopy(raw), words, display)
    admitted_box = next(b['bbox'] for b in admitted['blocks'] if b['id'] == 'graphic')
    assert admitted_box != graphic['bbox']
    controller = ur.Controller(SimpleNamespace(key=None), [0], tmp_path)
    controller.answers[0] = copy.deepcopy(raw)
    controller.admit_answer(0, admitted)
    saved = copy.deepcopy(controller.admitted_answers)
    controller.degraded[(0, 'graphic')] = dict(page=0, block='graphic')
    selected = controller.filter_answer(0, raw)
    assert next(b['bbox'] for b in selected['blocks'] if b['id'] == 'graphic') == admitted_box
    rebound, _ = vp.bind_geometry(selected, words, display)
    assert next(b['text'] for b in rebound['blocks'] if b['id'] == 'heading') == 'A clear sample heading'
    assert next(b['text'] for b in rebound['blocks'] if b['id'] == 'graphic') == 'Original source image'
    assert controller.admitted_answers == saved
    assert controller.answers[0] == raw
    controller.admitted_answers.clear()
    unproved = controller.filter_answer(0, raw)
    assert next(b['bbox'] for b in unproved['blocks'] if b['id'] == 'graphic') == graphic['bbox']
    with pytest.raises(ValueError, match='cannot be bound to printed pixels'):
        vp.bind_geometry(unproved, words, display)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('defect', ['disclosed', 'unmapped'])
def test_leading_art_cannot_join_across_a_disclosed_or_unmapped_body(side, defect):
    left = dict(id='previous-tail', role='body', source_regions=[[20, 500, 300, 550]], continues_next=True)
    barrier = dict(id='barrier', role='body', source_regions=[[20, 200, 300, 250]])
    selected = left if side == 'left' else barrier
    if defect == 'disclosed':
        selected['degraded_source_context'] = True
    else:
        selected['source_regions'] = []
    after = [dict(id='art', role='figure', degraded_source_context=True),
             dict(id='caption', role='caption'), barrier,
             dict(id='healthy', role='body', source_regions=[[20, 400, 300, 500]], continues_previous=False)]
    current = dict(body=['<figure>Original source image</figure>',
                         '<p class="caption">A clear caption.</p>',
                         '<p>A retained sample.</p>', '<p>A clear sample remains readable.</p>'])
    before = [left]
    saved = copy.deepcopy((before, after, current))
    assert vp._leading_art_continuation(dict(blocks=before), current, before, after,
                                       '<p>A clear previous sample continues</p>', after) is None
    assert (before, after, current) == saved

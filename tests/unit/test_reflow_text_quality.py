"""Synthetic text controls for per-page lexical and source-script admission."""
import copy
from pathlib import Path
import zipfile

import pymupdf
import pytest

from cps.services.reflow import assess, text_quality, vision_pipeline
from tests.unit.test_reflow_vision_pilot import SyntheticReader, answer, block

pytestmark = pytest.mark.unit
GOOD = 'Green baskets contain ripe apples while purple carts carry fresh bread.'
BAD = 'PaTchWOrk roOtIng blorple fliMmEd Zzxqv thAte'
LEXICON = frozenset(GOOD.lower().rstrip('.').split()) | {'patchwork', 'rooting'}


def test_internal_case_with_dictionary_misses_rejects_even_short_headings():
    result = text_quality.inspect(BAD, lexicon=LEXICON)
    assert result['needs_read']
    assert result['case_flip_words'] == 4
    assert result['dictionary_misses'] == 4
    assert 'internal_case_and_dictionary_misses' in result['reasons']
    assert not text_quality.inspect(GOOD.upper(), lexicon=LEXICON)['needs_read']
    assert not text_quality.inspect(GOOD.title(), lexicon=LEXICON)['needs_read']
    assert text_quality.inspect((GOOD+' ')*20+'\n'+BAD,lexicon=LEXICON)['needs_read']
    assert text_quality.inspect('coNtAiNeR',lexicon={'container'})['needs_read']
    assert not text_quality.inspect('BrightForms maker joins CedarWorks today',lexicon={'maker','joins','today'})['needs_read']


def test_dense_unrecognizable_lowercase_passage_cannot_hide_in_clean_page():
    damage = ' '.join('zqxv' + chr(97+n) for n in range(16))
    result = text_quality.inspect((GOOD+' ')*20 + damage, lexicon=LEXICON)
    assert result['dictionary_miss_rate'] < .10
    assert result['worst_window_miss_rate'] == 1
    assert result['needs_read']
    assert result['case_flip_words'] == 0


def test_non_latin_words_are_not_english_misses_and_lost_script_is_rejected():
    # Letter combinations are written for this test, not a quoted passage.
    original = 'αλφαβήτα γαμμαδέλτα καππαλάμδα'
    metric = text_quality.inspect(original, lexicon=LEXICON)
    assert metric['latin_words'] == 0 and not metric['needs_read']
    assert text_quality.script_mismatches(original, GOOD) == ['GREEK']
    assert text_quality.script_mismatches(original, original) == []
    assert text_quality.script_mismatches('α + β = γ', GOOD) == []


def test_every_page_and_its_actual_notes_are_swept_without_generated_disclosures():
    pages = [dict(pno=0,body=['<p>'+GOOD+'</p>'],asides=[]),
             dict(pno=1,body=['<h2>'+GOOD+'</h2>'],asides=['<aside>'+BAD+'</aside>'])]
    records = {0:dict(source='native'),1:dict(source='native')}
    sweep = text_quality.sweep_pages(pages, records, [0,1], lexicon=LEXICON)
    assert sweep['checked_pages'] == [0,1]
    assert sweep['pages'][0]['passed'] and not sweep['pages'][1]['passed']
    with pytest.raises(ValueError,match='Every source page'):
        text_quality.validate_sweep(sweep,[0,1])
    pages[1]['asides']=['<aside class="vision-disclosure">'+BAD+'</aside>']
    clean = text_quality.sweep_pages(pages,records,[0,1],lexicon=LEXICON)
    text_quality.validate_sweep(clean,[0,1])
    with pytest.raises(ValueError,match='exactly once'):
        text_quality.sweep_pages(pages[:1],records,[0,1],lexicon=LEXICON)
    incomplete=copy.deepcopy(clean);incomplete['pages'].pop(1)
    with pytest.raises(ValueError,match='Every source page'):
        text_quality.validate_sweep(incomplete,[0,1])


def test_missing_dictionary_cannot_certify_a_text_layer(monkeypatch):
    monkeypatch.setattr(text_quality,'english_lexicon',lambda:None)
    metric=text_quality.inspect(GOOD)
    assert metric['needs_read'] and not metric['dictionary_available']
    assert metric['reasons']==['dictionary_unavailable']


def test_bad_native_text_is_not_classified_as_clean(monkeypatch):
    monkeypatch.setattr(text_quality,'english_lexicon',lambda:LEXICON)
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=300)
        page.insert_text((30,60),BAD)
        assert not vision_pipeline.clean_native(page)
        raw = (GOOD+' ')*4 + BAD
        assert not assess.looks_like_prose((BAD+' ')*10)
        assert text_quality.inspect(raw)['case_flip_words'] == 4


def test_bad_cached_reading_retains_original_image_without_publishing_junk(tmp_path,monkeypatch):
    monkeypatch.setattr(text_quality,'english_lexicon',lambda:LEXICON)
    # Avoid depending on a platform OCR spelling: rejection precedes geometry.
    from cps.services.reflow import vision_pixels
    monkeypatch.setattr(vision_pixels,'words',lambda *a,**k:[])
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=300);page.insert_text((30,60),BAD)
        reader=SyntheticReader(answer(block(BAD)))
        destination=tmp_path/'quality.epub'
        report=vision_pipeline.convert(doc,destination,reader)
        assert reader.calls == 1
        assert report['pages'][0]['source']=='original_source'
        assert report['text_quality']['passed']
        assert report['text_quality']['checked_pages']==[0]
        with zipfile.ZipFile(destination) as z:
            body='\n'.join(z.read(n).decode() for n in z.namelist() if n.endswith('.xhtml'))
            assert BAD not in body
            assert 'View retained original source page' in body
            assert any(n.endswith('.jpg') for n in z.namelist())


def test_source_script_mismatch_rejects_cached_latin_reading(tmp_path,monkeypatch):
    monkeypatch.setattr(text_quality,'english_lexicon',lambda:LEXICON)
    from cps.services.reflow import vision_pixels
    monkeypatch.setattr(vision_pixels,'words',lambda *a,**k:[])
    monkeypatch.setattr(vision_pipeline,'clean_native',lambda *a,**k:False)
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=300)
        page.insert_font(fontname='greek',fontfile=None,fontbuffer=pymupdf.Font('helv').buffer)
        # A built-in Unicode font produces genuine non-Latin source codepoints.
        page.insert_text((30,60),'αλφαβήτα γαμμαδέλτα',fontname='greek')
        assert text_quality.script_counts(page.get_text())['GREEK'] >= 4
        reader=SyntheticReader(answer(block(GOOD)))
        report=vision_pipeline.convert(doc,tmp_path/'script.epub',reader)
        assert report['pages'][0]['source']=='original_source'
        assert any('missing_source_script_GREEK' in e['reason'] for e in report['unit_recovery_events'])


def test_cached_non_latin_reading_keeps_printed_script_in_published_text(tmp_path,monkeypatch):
    monkeypatch.setattr(text_quality,'english_lexicon',lambda:LEXICON)
    from cps.services.reflow import vision_pixels
    monkeypatch.setattr(vision_pipeline,'clean_native',lambda *a,**k:False)
    original='αλφαβήτα γαμμαδέλτα'
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=300)
        page.insert_font(fontname='greek',fontbuffer=pymupdf.Font('helv').buffer)
        page.insert_text((30,60),original,fontname='greek')
        optical=[dict(text=w[4],bbox=list(w[:4])) for w in page.get_text('words')]
        monkeypatch.setattr(vision_pixels,'words',lambda *a,**k:copy.deepcopy(optical))
        reader=SyntheticReader(answer(block(original)))
        destination=tmp_path/'correct-script.epub'
        report=vision_pipeline.convert(doc,destination,reader)
        assert report['pages'][0]['source']=='vision'
        assert report['text_quality']['passed']
        with zipfile.ZipFile(destination) as archive:
            from xml.etree import ElementTree as ET
            reading=' '.join(''.join(ET.fromstring(archive.read(name)).itertext())
                for name in archive.namelist() if name.endswith('.xhtml'))
            assert original in reading


def test_unreliable_layer_is_not_repeated_as_visible_disclosure_prose(tmp_path,monkeypatch):
    monkeypatch.setattr(text_quality,'english_lexicon',lambda:LEXICON)
    from cps.services.reflow import build_epub
    from cps.services.reflow.source_display import SourceDisplay
    source=GOOD.replace('baskets','baSkEtS').replace('purple','zzqxv')
    words=[dict(text=w,bbox=[20+n*25,30,43+n*25,42]) for n,w in enumerate(source.split())]
    package=build_epub._Package(str(tmp_path/'disclosure.epub'))
    with pymupdf.open() as doc:
        doc.new_page(width=400,height=300)
        try:
            page,report=vision_pipeline.compile_page(answer(block(GOOD)),words,SourceDisplay(doc,0),0,package,{})
            assert report['source_quote_suppressions']
            html='\n'.join(page['body']+page['asides'])
            assert 'baSkEtS' not in html and 'zzqxv' not in html
            assert 'text layer is unreliable' in html
            assert 'Original printed reading' in html
            assert GOOD in text_quality.reading_text(page)
        finally:
            package.abandon()


def test_english_optical_fallback_cannot_erase_a_reader_non_latin_passage(tmp_path,monkeypatch):
    monkeypatch.setattr(text_quality,'english_lexicon',lambda:LEXICON)
    from cps.services.reflow import build_epub
    from cps.services.reflow.source_display import SourceDisplay
    original=' '.join(['αλφαβήτα','γαμμαδέλτα']*6)
    wrong=' '.join(['blorple','zzqxv']*6)
    optical=[dict(text=w,bbox=[20+n*25,30,43+n*25,42]) for n,w in enumerate(wrong.split())]
    package=build_epub._Package(str(tmp_path/'fallback-script.epub'))
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=300)
        page.insert_font(fontname='greek',fontbuffer=pymupdf.Font('helv').buffer)
        page.insert_textbox(pymupdf.Rect(20,20,380,180),original,fontname='greek',fontsize=9)
        assert text_quality.script_counts(page.get_text())['GREEK'] >= 4
        try:
            with pytest.raises(ValueError,match='lost during alignment: GREEK'):
                vision_pipeline.compile_page(answer(block(original)),optical,SourceDisplay(doc,0),0,package,{})
        finally:
            package.abandon()

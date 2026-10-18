"""Synthetic transcription alternatives and occurrence geometry written here."""
import pytest
from cps.services.reflow.pixel_adjudication import adjudicate

pytestmark=pytest.mark.unit


def words(text, confidence=99):
    return [dict(text=t,bbox=[i*20,20,i*20+18,32],confidence=confidence)
            for i,t in enumerate(text.split())]


def test_pixel_read_restores_short_model_deletion():
    source=words('We carried the blue box to the open door today.')
    result=adjudicate('We carried the open door today.'.split(),source,source)
    assert result['tokens']==[w['text'] for w in source]
    assert result['decisions'][0]['status']=='RESOLVED'
    assert result['decisions'][0]['chosen']==['the','blue','box','to']
    assert result['marked']


def test_pixel_read_rejects_inserted_real_word():
    source=words('The clear sample remains readable today.')
    result=adjudicate('The clear sample never remains readable today.'.split(),source,source,
        absence_proof=lambda left,right:True)
    assert result['tokens']==[w['text'] for w in source]
    assert result['decisions'][0]['chosen']==[]
    assert result['decisions'][0]['status']=='RESOLVED'
    assert result['marked']


def test_corroborated_model_correction_stays_marked_word_by_word():
    source=words('The clear sample remalns readable today.')
    result=adjudicate('The clear sample remains readable today.'.split(),source,
                      words('The clear sample remains readable today.'))
    assert result['tokens'][3]=='remains'
    assert result['marked']=={3}
    assert result['word_notes'][3]['source']==['remalns']
    assert result['word_notes'][3]['chosen']==['remains']
    assert result['word_notes'][3]['status']=='RESOLVED'


def test_repeat_without_unique_pixel_context_cannot_settle_edit():
    source=words('The sample remalns readable today.')
    witness=words('The sample remains readable today. The sample remains readable today.')
    result=adjudicate('The sample remains readable today.'.split(),source,witness)
    assert result['word_notes'][2]['status']=='UNRESOLVED'
    assert result['marked']=={2}


def test_roman_numeral_rewrite_requires_pixel_agreement():
    source=words('Please read section III before opening the box.')
    result=adjudicate('Please read section 111 before opening the box.'.split(),source,source)
    assert result['tokens'][3]=='III'
    assert result['word_notes'][3]['status']=='RESOLVED'


def test_low_confidence_witness_cannot_certify_a_word():
    source=words('The sample remalns readable today.')
    result=adjudicate('The sample remains readable today.'.split(),source,
                      words('The sample remains readable today.',30))
    assert result['word_notes'][2]['status']=='UNRESOLVED'


def test_real_conversion_restores_model_deleted_words(tmp_path,monkeypatch):
    import shutil
    if not shutil.which('tesseract'):pytest.skip('requires real OCR')
    from tests.unit import test_round4_defects as ink
    page=ink.InkPage()
    printed='We carried the blue box to the open door today.'
    page.text(printed,'body',model='We carried the open door today.')
    _,report,docs=ink.convert_ink(tmp_path,monkeypatch,[page])
    assert ink.body_texts(docs)==[printed]
    decisions=report['pages'][0]['blocks'][0]['adjudication']
    assert any(d['status']=='RESOLVED' and d['chosen']!=d['model'] for d in decisions)
    disclosures=[n for root in docs.values() for n in root.iter()
                 if n.get('class')=='footnote vision-disclosure']
    assert len(disclosures)==4
    assert any('Pixel-verified correction' in ''.join(n.itertext()) for n in disclosures)
    assert all(printed not in ''.join(n.itertext()) for n in disclosures)


def test_fallback_pixel_read_repairs_ocr_misspelling_and_omitted_pronoun():
    from cps.services.reflow.pixel_adjudication import refine_fallback
    before='After setding on the bench we read the clear sample today.'
    printed='After settling on the bench we I read the clear sample today.'
    lexicon=set(printed.lower().replace('.','').split())
    result=refine_fallback(before.split(),words(printed),lexicon,source=words(printed))
    assert result['tokens']==printed.split()
    assert len([d for d in result['decisions'] if d['status']=='RESOLVED'])==2
    assert result['marked']=={1,6}


def test_dictionary_cannot_delete_a_real_word_or_rewrite_an_unfamiliar_name():
    from cps.services.reflow.pixel_adjudication import refine_fallback
    text='We saw Brindle under the clear bridge on Tuesday.'
    result=refine_fallback(text.split(),words('We saw bundle under the bridge on Tuesday.'),{'bundle','the'},source=words(text))
    assert result['tokens']==text.split()
    assert all(d['status']=='UNRESOLVED' for d in result['decisions'])


def test_multiple_note_references_keep_their_own_lexical_boundaries():
    text='Clear <n:first> words <n:second> remain readable.'
    result=adjudicate(text.split(),words('Clear words remain readable.'),
                      words('Clear words remain readable.'))
    assert result['tokens']==text.split()


def test_agreeing_words_do_not_erase_unresolved_reader_uncertainty():
    text='The clear sample remains readable today.'
    result=adjudicate(text.split(),words(text),words(text,30),uncertain={3})
    assert result['marked']=={3}
    assert result['word_notes'][3]['status']=='UNRESOLVED'


def test_unique_spelling_in_a_second_copy_cannot_settle_the_first_occurrence():
    source=words('The sample remains clear today.')
    witness=words('The sample rernains clear today. The sample becomes clear today.')
    result=adjudicate('The sample becomes clear today.'.split(),source,witness)
    assert result['word_notes'][2]['status']=='UNRESOLVED'
    assert result['boxes'][2]==source[2]['bbox']


def test_matching_layer_and_first_read_do_not_hide_a_pixel_conflict():
    from cps.services.reflow.pixel_adjudication import audit_agreement
    original=words('The sample remalns readable today.')
    witness=words('The sample remains readable today.')
    notes=audit_agreement([w['text'] for w in original],original,witness)
    assert set(notes)=={2}
    assert notes[2]['status']=='UNRESOLVED'
    assert notes[2]['alternative']==['remains']


def test_fragment_binding_keeps_repeated_words_in_their_own_positions():
    from cps.services.reflow.pixel_adjudication import fragment_word_ranges
    fragment='<p>Clear sample then another clear sample.</p>'
    tokens='Clear sample then another clear sample.'.split()
    ranges=fragment_word_ranges(fragment,tokens)
    assert ranges is not None
    assert fragment[ranges[4][0]:ranges[4][1]]=='clear'
    assert ranges[4][0]>ranges[0][1]
    assert fragment_word_ranges(fragment,['different']) is None


def test_native_layer_pixel_conflict_is_visible_at_its_own_occurrence(tmp_path,monkeypatch):
    import pymupdf
    from types import SimpleNamespace
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from tests.unit import test_round4_defects as ink
    text='The clear sample remalns readable beside the window today.'
    with pymupdf.open()as doc:
        page=doc.new_page(width=400,height=500)
        page.insert_text((30,70),text,fontsize=10)
        original=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        witness=[dict(w,text='remains' if w['text']=='remalns'else w['text'],confidence=88)for w in original]
        monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
        reader=SimpleNamespace(cache=tmp_path/'cache'/'readings',key=None)
        output=tmp_path/'native.epub'
        result=vp.convert(doc,str(output),reader)
    record=result['pages'][0]
    assert record['source']=='native'
    decisions=[d for b in record['blocks']for d in b.get('adjudication',[])]
    assert any(d.get('alternative')==['remains']for d in decisions)
    import zipfile
    with zipfile.ZipFile(output)as archive:
        html=''.join(archive.read(n).decode()for n in archive.namelist()if n.endswith('.xhtml'))
    assert 'Unresolved source reading: remalns.' in html
    assert 'class="source-disclosure-ref"' in html
    assert 'images/v4-native-d-' in html


def test_unique_high_confidence_pixel_agreement_settles_unchanged_uncertain_word():
    text='The clear sample remains readable today.'
    result=adjudicate(text.split(),words(text),words(text),uncertain={3})
    assert result['tokens']==text.split()
    assert result['marked']==set()
    assert result['verified_unchanged']==[3]


def test_native_ocr_misspelling_is_corrected_only_with_contextual_pixel_support(tmp_path,monkeypatch):
    import pymupdf,zipfile
    from types import SimpleNamespace
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    text='The clear sample remalns readable beside the window today.'
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=500)
        page.insert_text((30,70),text,fontsize=10)
        witness=[dict(text='remains' if w[4]=='remalns' else w[4],bbox=list(w[:4]),confidence=99)
                 for w in page.get_text('words')]
        monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
        output=tmp_path/'native.epub'
        report=vp.convert(doc,str(output),SimpleNamespace(cache=tmp_path/'cache/readings',key=None))
    with zipfile.ZipFile(output) as archive:
        html=''.join(archive.read(n).decode()for n in archive.namelist()if n.endswith('.xhtml'))
    assert 'sample remains' in html
    assert 'Pixel-verified correction: remalns → remains.' in html
    assert any(d['status']=='RESOLVED'for b in report['pages'][0]['blocks']for d in b.get('adjudication',[]))


def test_fallback_cannot_borrow_the_clear_word_from_another_occurrence():
    from cps.services.reflow.pixel_adjudication import refine_fallback
    text='The sample remalns clear today.'
    result=refine_fallback(text.split(),words('The sarnple rernains clear today. The sample remains clear today.'),
        {'the','sample','remains','clear','today'},source=words(text))
    assert result['tokens']==text.split()
    assert all(d['status']=='UNRESOLVED'for d in result['decisions'])


def test_adjacent_disjoint_row_cannot_certify_the_source_occurrence():
    from cps.services.reflow.pixel_adjudication import _same_occurrence
    assert not _same_occurrence(dict(bbox=[0,0,30,14]),dict(bbox=[0,16,30,22]))


def test_native_factory_joined_wrap_still_receives_an_independent_pixel_read(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=500)
        page.insert_text((30,70),'The sam-',fontsize=10)
        page.insert_text((30,85),'ple remains clear today.',fontsize=10)
        physical=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        regions=[vp.vision_reading.union([w['bbox']for w in physical[:2]]),
                 vp.vision_reading.union([w['bbox']for w in physical[2:]])]
        seen=[]
        def witness(*args):
            seen.append(args[1]);return physical
        monkeypatch.setattr(pa,'read_crop',witness)
        records=[dict(id='wrapped',role='body',source_regions=regions)]
        published=dict(pno=0,body=['<p>The sample remains clear today.</p>'],asides=[])
        vp._audit_native_pixels(doc,published,records,tmp_path,object(),{})
        assert seen==[regions]
        assert published['body']==['<p>The sample remains clear today.</p>']
        assert records[0]['published_tokens']=='The sample remains clear today.'.split()


def test_punctuation_mark_does_not_reclassify_a_source_note_as_garbled(tmp_path):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    text='Brindle Wembury, Clatterford Pemberley, 9-10.'
    source=words(text)
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=300,height=400)
        page,report=vp.compile_page({'blocks':[block(text.replace('9-10','9–10'),
            role='footnote',marker='3',bbox=(0,0,1000,1000))]},source,SourceDisplay(doc,0),0,Package(),{},pixel_words=source)
    assert 'Brindle Wembury, Clatterford Pemberley,' in page['asides'][0]
    assert report['marks']==1
    assert '9–10.' in page['asides'][0]


def test_local_numeral_correction_preserves_original_occurrence_and_uncertainty(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    text='Please read section 11 before opening the box.'
    original=words(text)
    witness=[dict(w,text='III' if w['text']=='11' else w['text'])for w in original]
    monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=300,height=400)
        page,report=vp.compile_page({'blocks':[block(text,role='footnote',marker='2',bbox=(0,0,1000,1000))]},
            [],SourceDisplay(doc,0),0,Package(),{},pixel_words=original,local_ocr=True,
            local_ocr_units=[dict(id='b1',words=original,source_word_indices=list(range(len(original))))],
            note_context={'cache':tmp_path/'cache'})
    assert report['blocks'][0]['published_tokens'][3]=='III'
    assert report['local_ocr_units'][0]['words'][3]['text']=='11'
    assert 'class="uncertain"' in page['asides'][0]
    assert any('11 → III.' in aside for aside in page['asides'][1:])
    assert any(d['status']=='RESOLVED'for d in report['blocks'][0]['adjudication'])


def test_model_formatting_tags_are_not_published_as_printed_words(tmp_path):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    original=words('Please read the clear sample today.')
    answer={'blocks':[block('Please read <i>the clear sample</i> today.',bbox=(0,0,1000,1000))]}
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=300,height=400)
        page,report=vp.compile_page(answer,original,SourceDisplay(doc,0),0,Package(),{},pixel_words=original)
    assert report['blocks'][0]['published_tokens']=='Please read the clear sample today.'.split()
    assert '&lt;i&gt;' not in ''.join(page['body'])
    assert answer['blocks'][0]['text']=='Please read <i>the clear sample</i> today.'
    assert '<i>the' in report['blocks'][0]['original_model_tokens']


def test_unproved_model_deletion_keeps_source_words_and_a_word_level_warning():
    text='The clear sample carries Brindle beside the window today.'
    result=adjudicate('The clear sample carries beside the window today.'.split(),words(text),words(text,30))
    assert result['tokens']==text.split()
    assert result['word_notes'][4]['status']=='UNRESOLVED'
    assert result['word_notes'][4]['chosen']==['Brindle']
    assert result['boxes'][4]==words(text)[4]['bbox']


def test_missing_ocr_word_cannot_prove_an_insertion_across_disjoint_source_crops():
    text='The clear sample carries Brindle beside the window today.'
    source=words(text)
    witness=[w for w in source if w['text']!='Brindle']
    result=adjudicate(text.split(),witness,witness)
    assert result['tokens']==text.split()
    assert result['word_notes'][4]['status']=='UNRESOLVED'


def test_blank_evidence_checks_intervening_ink_and_same_crop():
    import pymupdf
    from cps.services.reflow.source_display import SourceDisplay
    from cps.services.reflow.pixel_adjudication import blank_between
    with pymupdf.open() as doc:
        page=doc.new_page(width=300,height=100)
        page.insert_text((80,40),'Brindle',fontsize=12)
        display=SourceDisplay(doc,0)
        left=dict(bbox=[30,25,60,45]);right=dict(bbox=[140,25,170,45])
        assert not blank_between(display,left,right,[[30,25,60,45],[140,25,170,45]])
        assert not blank_between(display,left,right,[[30,25,170,45]])
        assert not blank_between(display,dict(bbox=[30,60,60,80]),dict(bbox=[140,60,170,80]),[[30,60,60,80],[140,60,170,80]])
        assert blank_between(display,dict(bbox=[30,60,60,80]),dict(bbox=[140,60,170,80]),[[30,60,170,80]])


def test_word_correction_note_explains_evidence_and_keeps_its_source_owner(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa,text_quality
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    text='We carried the rare Brindlc sample beside the quiet window today.'
    original=words(text)
    monkeypatch.setattr(pa,'read_crop',lambda *args:[dict(w,text='Brindle' if w['text']=='Brindlc' else w['text'])for w in original])
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=400,height=400)
        page,report=vp.compile_page({'blocks':[block(text.replace('Brindlc','Brindle'),role='footnote',marker='2',bbox=(0,0,1000,1000))]},
            original,SourceDisplay(doc,0),0,Package(),{},pixel_words=original,note_context={'cache':tmp_path/'cache'})
    assert any('Brindlc → Brindle.' in aside for aside in page['asides'])
    report['source']='vision'
    assert vp._compiled_text_quality([page],{0:report},[0])['passed']
    parts=text_quality.publication_parts(page)
    correction=next(p for p in parts if 'Brindlc → Brindle.' in p['text'])
    assert correction['owners']==[dict(page=0,block='b1')]
    assert 'An independent upright crop supports this correction.' in correction['text']


def test_fallback_adjudication_retains_the_first_reader_alternative(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    text='Please read the clear small wooden bridge beside the quiet stream before opening the box today.'
    witness=words(text)
    baseline=witness[:3]+witness[11:]
    monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=400)
        page,report=vp.compile_page({'blocks':[block(text,role='footnote',marker='2',bbox=(0,0,1000,1000))]},
            baseline,SourceDisplay(doc,0),0,Package(),{},pixel_words=baseline,note_context={'cache':tmp_path/'cache'})
    assert report['blocks'][0]['published_tokens']==text.split()
    decisions=report['blocks'][0]['adjudication']
    assert any(d['model']==text.split()[3:11] and d['chosen']==d['model'] and d['status']=='RESOLVED'for d in decisions)


def test_unplaced_multiword_edit_has_bounded_word_notes_not_repeated_paragraphs(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    original=words('Please carry the blue box beside the quiet window today.')
    model='Please carry the small wooden sample and open box beside the quiet window today.'
    monkeypatch.setattr(pa,'read_crop',lambda *args:words(model,30))
    result=adjudicate(model.split(),original,words(model,30))
    assert all(len(note['chosen'])==1 for note in result['word_notes'].values())
    assert any(note.get('mapping')=='unplaced_within_span'for note in result['word_notes'].values())
    assert any(len(d['model'])>1 for d in result['decisions'])
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=400)
        page,report=vp.compile_page({'blocks':[block(model,role='footnote',marker='2',bbox=(0,0,1000,1000))]},
            original,SourceDisplay(doc,0),0,Package(),{},pixel_words=original,note_context={'cache':tmp_path/'cache'})
    assert report['blocks'][0]['published_tokens']==model.split()
    notes=[aside for aside in page['asides'] if 'Unresolved source reading' in aside]
    assert len(notes)>=5
    assert all('small wooden sample and open' not in aside for aside in notes)
    assert any('source-word correspondence is unresolved' in aside for aside in notes)


@pytest.mark.parametrize('kind',['word','label','label-punctuation','glyphs'])
def test_native_emission_order_does_not_duplicate_physical_words(tmp_path,monkeypatch,kind):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    text='Please carry the clear sample beside our quiet window today.'
    physical=[dict(w,bbox=[w['bbox'][0]+20,w['bbox'][1],w['bbox'][2]+20,w['bbox'][3]])for w in words(text)]
    if kind=='word':native=physical[:5]+physical[6:]+physical[5:6]
    elif kind.startswith('label'):
        native=physical+[dict(text='2}' if kind=='label-punctuation' else '2',bbox=[2,20,10,32])]
    else:
        native=physical[:3]+[dict(text='cl',bbox=[80,20,86,32])]+physical[4:]+[
            dict(text='ea',bbox=[86,20,93,32]),dict(text='r',bbox=[93,20,98,32])]
    monkeypatch.setattr(pa,'read_crop',lambda *args:physical)
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=400)
        page,report=vp.compile_page({'blocks':[block(text,role='footnote',marker='2',bbox=(0,0,1000,1000))]},
            native,SourceDisplay(doc,0),0,Package(),{},pixel_words=physical,note_context={'cache':tmp_path/'cache'})
    assert report['blocks'][0]['published_tokens']==text.split()
    assert page['asides'][0].count('<p>2 ')==1


def test_reader_formatting_metadata_preserves_the_uncertain_word_index():
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    original=words('Please read the clear sample today.')
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=400,height=400)
        _,report=vp.compile_page({'blocks':[block('Please <i> read </i> the clear sample today.',low=[2],bbox=(0,0,1000,1000))]},
            [],SourceDisplay(doc,0),0,Package(),{},pixel_words=original)
    assert report['blocks'][0]['published_tokens']==[w['text']for w in original]
    assert report['blocks'][0]['marked']==[1]

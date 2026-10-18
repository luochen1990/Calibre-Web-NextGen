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


def test_fragment_binding_keeps_inline_ordinals_and_typed_references_intact():
    from cps.services.reflow.pixel_adjudication import fragment_word_ranges
    fragment='<p>The 1<sup>st</sup> clear sample has a printed reference<a class="noteref" href="#note"><sup>7</sup></a>.</p>'
    tokens='The 1st clear sample has a printed reference7.'.split()
    ranges=fragment_word_ranges(fragment,tokens)
    assert ranges is not None
    assert fragment[slice(*ranges[2])]=='clear'
    assert fragment[slice(*ranges[1])]=='1<sup>st</sup>'
    assert fragment[slice(*ranges[7])]=='reference<a class="noteref" href="#note"><sup>7</sup></a>.'
    assert fragment_word_ranges(fragment,tokens[:-1]+['reference8.']) is None


def test_native_conflict_next_to_inline_ordinal_keeps_live_text_and_style(tmp_path,monkeypatch):
    import re
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    text='The 1st clear sample remalns readable today.'
    fragment='<p>The 1<sup>st</sup> clear sample remalns readable today.</p>'
    class Package:
        def disclosure_image(self,*args):pass
    with pymupdf.open()as doc:
        leaf=doc.new_page(width=400,height=500)
        leaf.insert_text((30,70),text,fontsize=10)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=88)for w in leaf.get_text('words')]
        witness=[dict(w,text='remains'if w['text']=='remalns'else w['text'])for w in source]
        monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
        page=dict(pno=0,body=[fragment],asides=[])
        records=[dict(id='native-0',role='body',source_regions=[[25,55,390,80]])]
        vp._audit_native_pixels(doc,page,records,tmp_path,Package(),{})
    assert '1<sup>st</sup>'in page['body'][0]
    assert page['body'][0].count('class="source-disclosure-ref"')==1
    authored=re.sub(r'<a[^>]*class="source-disclosure-ref"[^>]*>.*?</a>','',page['body'][0])
    assert authored==fragment,'only the owned generated control is excluded from authored text'
    assert len(page['asides'])==1 and 'Unresolved source reading: remalns.'in page['asides'][0]
    assert records[0]['marked']==[4]


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


def test_local_heading_case_and_name_can_follow_two_same_occurrence_pixel_reads():
    from cps.services.reflow.pixel_adjudication import refine_fallback
    original=words('The cLeAr sAmPlE FROM Brindlc')
    witness=words('THE CLEAR SAMPLE FROM Brindle')
    result=refine_fallback([w['text'] for w in original],witness,
        {'the','clear','sample','from'},source=original,corroboration=witness)
    assert result['tokens']==[w['text'] for w in witness]
    assert all(d['status']=='RESOLVED' for d in result['decisions'])
    assert result['marked']


def test_local_unknown_name_cannot_follow_a_different_occurrence_pixel_read():
    from cps.services.reflow.pixel_adjudication import refine_fallback
    original=words('The clear sample from Brindlc remains here today.')
    witness=words('The clear sample from Brindle remains here today.')
    remote=[dict(w,bbox=[x+400 for x in w['bbox']]) for w in witness]
    result=refine_fallback([w['text'] for w in original],witness,
        {'the','clear','sample','from','remains','here','today'},source=original,corroboration=remote)
    assert result['tokens']==[w['text'] for w in original]
    assert any(d['status']=='UNRESOLVED' for d in result['decisions'])


def test_local_two_view_name_repair_is_live_but_keeps_original_words_and_uncertainty(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    text='The clear sample from Brindlc remains here today.'
    original=words(text)
    witness=[dict(w,text='Brindle' if w['text']=='Brindlc' else w['text'])for w in original]
    monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=400,height=400)
        page,report=vp.compile_page({'blocks':[block(text,role='footnote',marker='2',bbox=(0,0,1000,1000))]},
            [],SourceDisplay(doc,0),0,Package(),{},pixel_words=witness,local_ocr=True,
            local_ocr_units=[dict(id='b1',words=original,source_word_indices=list(range(len(original))))],
            note_context={'cache':tmp_path/'cache'})
    assert report['blocks'][0]['published_tokens'][4]=='Brindle'
    assert report['local_ocr_units'][0]['words'][4]['text']=='Brindlc'
    assert 'class="uncertain"' in page['asides'][0]
    assert any('Brindlc → Brindle.' in aside for aside in page['asides'][1:])


def test_one_word_pixel_choice_survives_a_neighbouring_punctuation_conflict():
    source=words('The clear sample carries Brindle beside the window today.')
    witness=[dict(w,text='beside,' if w['text']=='beside' else w['text'])for w in source]
    result=adjudicate('The clear sample carries Brindlc beside the window today.'.split(),source,witness)
    assert result['tokens'][4]=='Brindle'
    assert result['word_notes'][4]['status']=='RESOLVED'
    assert result['word_notes'][4]['chosen']==['Brindle']
    assert 4 in result['marked']


def test_one_confident_word_is_settled_without_certifying_its_damaged_neighbour():
    source=words('Please read Brindle Calleton beside the quiet window today.')
    witness=[dict(w,confidence=30 if w['text']=='Calleton' else 99)for w in source]
    result=adjudicate('Please read Brindlc Calletqn beside the quiet window today.'.split(),source,witness)
    assert result['tokens'][2:4]==['Brindle','Calletqn']
    assert result['word_notes'][2]['status']=='RESOLVED'
    assert result['word_notes'][3]['status']=='UNRESOLVED'


def test_two_pixel_views_can_settle_a_third_spelling_at_the_original_occurrence():
    source=words('Please read Brindlc beside the quiet window today.')
    witness=[dict(w,text='Brindle' if w['text']=='Brindlc' else w['text'])for w in source]
    model='Please read Brindlq beside the quiet window today.'.split()
    unproved=adjudicate(model,source,witness)
    result=adjudicate(model,source,witness,corroboration=witness)
    assert unproved['tokens'][2]=='Brindlq'
    assert result['tokens'][2]=='Brindle'
    assert result['word_notes'][2]['status']=='RESOLVED'
    assert result['word_notes'][2]['source']==['Brindlc']
    assert result['word_notes'][2]['model']==['Brindlq']


def test_pixel_word_boundary_survives_an_adjacent_unresolved_punctuation_read():
    source=words('Please read clear bridge beside the quiet window today.')
    witness=[dict(w,text='beside,' if w['text']=='beside' else w['text'])for w in source]
    result=adjudicate('Please read clearbridge beside the quiet window today.'.split(),source,witness)
    assert result['tokens'][2:4]==['clear','bridge']
    assert all(i in result['marked'] for i in (2,3))
    assert any(d['status']=='RESOLVED' and d['chosen']==['clear','bridge']for d in result['decisions'])


def test_native_unknown_spelling_uses_two_pixel_reads_and_remains_marked(tmp_path,monkeypatch):
    import pymupdf,zipfile
    from types import SimpleNamespace
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa,vision_pixels
    text='The clear sample from Brindlc remains beside the quiet window today.'
    with pymupdf.open()as doc:
        page=doc.new_page(width=500,height=500)
        page.insert_text((30,70),text,fontsize=10)
        original=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        witness=[dict(w,text='Brindle' if w['text']=='Brindlc'else w['text'])for w in original]
        monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
        monkeypatch.setattr(vision_pixels,'words',lambda *args:witness)
        output=tmp_path/'native.epub'
        result=vp.convert(doc,str(output),SimpleNamespace(cache=tmp_path/'cache'/'readings',key=None))
    assert result['pages'][0]['source']=='native'
    assert any(d['status']=='RESOLVED' and d['chosen']==['Brindle']
               for b in result['pages'][0]['blocks']for d in b.get('adjudication',[]))
    with zipfile.ZipFile(output)as z:
        html=''.join(z.read(n).decode()for n in z.namelist()if n.endswith('.xhtml'))
    assert 'Brindlc → Brindle.' in html
    assert 'Pixel-verified correction:' in html


@pytest.mark.parametrize('break_char',['\u00ad','-'])
def test_pixel_word_boundary_does_not_split_a_printed_wrap(break_char):
    source=words('Please read clear'+break_char+' bridge beside the quiet window today.')
    witness=[dict(w,text=w['text'].rstrip('\u00ad-'))for w in source]
    result=adjudicate('Please read clearbridge beside the quiet window today.'.split(),source,witness)
    assert result['tokens'][2]=='clearbridge'
    assert result['tokens'][3]=='beside'


@pytest.mark.parametrize('accented',['cafélane','cafe\u0301lane'])
def test_pixel_word_boundary_keeps_the_original_model_diacritics(accented):
    source=words('Please read cafe lane beside the quiet window today.')
    result=adjudicate(['Please','read',accented,'beside','the','quiet','window','today.'],source,source)
    assert result['tokens'][2:4]==[accented[:-4],'lane']
    assert all(i in result['marked']for i in (2,3))
    assert any(d['status']=='UNRESOLVED'for d in result['decisions'])


@pytest.mark.parametrize('pixel',['cafe','cafè'])
def test_latin_pixel_alternative_cannot_certify_a_different_accent(pixel):
    source=words('Please read cafe beside the quiet window today.')
    witness=[dict(w,text=pixel if w['text']=='cafe'else w['text'])for w in source]
    result=adjudicate('Please read café beside the quiet window today.'.split(),source,witness,corroboration=witness)
    assert result['tokens'][2]=='café'
    assert result['word_notes'][2]['status']=='UNRESOLVED'


def test_pixel_word_boundary_requires_one_physical_row():
    source=words('Please read clear bridge beside the quiet window today.')
    source[3]['bbox']=[0,40,18,52]
    result=adjudicate('Please read clearbridge beside the quiet window today.'.split(),source,source)
    assert result['tokens'][2]=='clearbridge'
    assert result['tokens'][3]=='beside'


def test_local_two_pixel_views_cannot_certify_accent_absence():
    from cps.services.reflow.pixel_adjudication import refine_fallback
    source=words('Please read café beside the quiet window today.')
    witness=[dict(w,text='cafe'if w['text']=='café'else w['text'])for w in source]
    result=refine_fallback([w['text']for w in source],witness,{'read','cafe'},source=source,corroboration=witness)
    assert result['tokens'][2]=='café'
    assert result['word_notes'][2]['status']=='UNRESOLVED'


def test_native_footnote_receives_its_own_independent_pixel_audit(tmp_path,monkeypatch):
    import pymupdf,zipfile
    from types import SimpleNamespace
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    with pymupdf.open()as doc:
        page=doc.new_page(width=500,height=500)
        page.insert_text((30,70),'The clear sample remains readable beside the quiet window today.',fontsize=11)
        page.insert_text((355,65),'1',fontsize=7)
        page.insert_text((30,440),'1 The clear sample remalns readable beside the quiet window today.',fontsize=8)
        original=[dict(text=w[4],bbox=list(w[:4]),confidence=88)for w in page.get_text('words')]
        witness=[dict(w,text='remains'if w['text']=='remalns'else w['text'])for w in original]
        monkeypatch.setattr(pa,'read_crop',lambda display,regions,cache:[w for w in witness if any(vp.owned_words([w],b)for b in regions)])
        output=tmp_path/'native-note.epub'
        result=vp.convert(doc,str(output),SimpleNamespace(cache=tmp_path/'cache'/'readings',key=None))
    assert result['pages'][0]['source']=='native'
    with zipfile.ZipFile(output)as z:
        html=''.join(z.read(n).decode()for n in z.namelist()if n.endswith('.xhtml'))
    assert 'class="footnote"' in html
    assert 'Unresolved source reading: remalns.' in html
    assert any(d.get('alternative')==['remains']for b in result['pages'][0].get('native_note_adjudication',[])for d in b.get('adjudication',[]))


def test_native_note_quality_failure_keeps_its_registered_note_owner(monkeypatch):
    from types import SimpleNamespace
    from cps.services.reflow import vision_pipeline as vp,unit_recovery
    monkeypatch.setattr(unit_recovery,'current',lambda:SimpleNamespace(check_units=lambda units:None,fragment_owners=lambda fragment:[dict(page=0,block='native-0')]))
    good='The clear sample remains readable beside the quiet window today. '*20
    bad=' '.join('qxv'+str(i)+'zrt'for i in range(50))
    page=dict(pno=0,body=['<p><span data-source-page="0" data-source-block="native-0">'+good+'</span></p>'],asides=['<aside class="footnote"><p><span data-source-page="0" data-source-block="native-note-0">'+bad+'</span></p></aside>'])
    records={0:dict(source='native',blocks=[dict(id='native-0',role='body')],native_note_adjudication=[dict(id='native-note-0',role='footnote')])}
    with pytest.raises(unit_recovery.UnitFailure)as error:vp._compiled_text_quality([page],records,[0])
    assert error.value.units==[dict(page=0,block='native-note-0')]


@pytest.mark.parametrize('empty_box',[False,True])
def test_native_continued_note_binds_unique_text_without_changing_quotes(tmp_path,monkeypatch,empty_box):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa,assemble,build_epub
    text='The reader\'s clear sample says "readable" beside the quiet window.'
    with pymupdf.open()as doc:
        page=doc.new_page(width=500,height=500);page.insert_text((30,70),text,fontsize=10)
        witness=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        box=(0,0,0,0)if empty_box else vp.vision_reading.union([w['bbox']for w in witness])
        note=assemble.Note(num=None,text=text,pno=0,bbox=box)
        book=assemble.Book(notes=[note],pages={0:[]})
        fragment=build_epub._aside_html(note,{},set())
        page_data=dict(pno=0,body=[],asides=[fragment])
        monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
        result=vp._audit_native_note_pixels(doc,book,page_data,tmp_path/'cache',None,{})
    assert len(result)==1 and result[0]['published_tokens']==text.split()
    assert vp._seam_text(page_data['asides'][0])==text
    assert 'data-source-block="native-note-0"'in page_data['asides'][0]


def test_native_mixed_glyph_note_cannot_bypass_live_word_audit(tmp_path):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,assemble
    with pymupdf.open()as doc:
        doc.new_page(width=300,height=300)
        note=assemble.Note(num=1,text='Ordinary sample A ends.',pno=0,bbox=(20,20,220,50),glyph_fallback=True,
            glyph_runs=[('t','Ordinary sample '),('glyph','A'),('t',' ends.')])
        with pytest.raises(ValueError,match='glyph'):
            vp._audit_native_note_pixels(doc,assemble.Book(notes=[note]),dict(pno=0,body=[],asides=[]),tmp_path,None,{})


def test_native_note_word_equal_to_label_binds_only_its_lexical_copy(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa,assemble,build_epub
    with pymupdf.open()as doc:
        page=doc.new_page(width=300,height=300);page.insert_text((30,70),'1 1',fontsize=10)
        witness=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        note=assemble.Note(num=1,text='1',pno=0,bbox=vp.vision_reading.union([w['bbox']for w in witness]))
        fragment=build_epub._scope_ids(build_epub._aside_html(note,{'1':'ref'}, {'1'}),0)
        page_data=dict(pno=0,body=[],asides=[fragment])
        monkeypatch.setattr(pa,'read_crop',lambda display,regions,cache:[w for w in witness if any(vp.owned_words([w],b)for b in regions)])
        result=vp._audit_native_note_pixels(doc,assemble.Book(notes=[note]),page_data,tmp_path,None,{})
    assert len(result)==1 and result[0]['published_tokens']==['1']
    assert '<a class="note-return" href="#ref" aria-label="Return to note reference">1</a>'in page_data['asides'][0]
    import json
    from xml.etree import ElementTree
    tree=ElementTree.fromstring('<root xmlns:epub="http://www.idpf.org/2007/ops">'+page_data['asides'][0]+'</root>')
    spans=[node for node in tree.iter()if node.get('data-source-block')=='native-note-0']
    assert len(spans)==1 and ''.join(spans[0].itertext())=='1'
    assert json.loads(spans[0].get('data-source-regions'))==[witness[1]['bbox']]


def test_inconclusive_second_read_cannot_restore_rejected_model_paragraph(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    text='The genuine printed paragraph follows quiet rivers past ordinary villages before sunset.'
    invented='Imaginary violet machines announce impossible discoveries beneath a fictional palace every evening.'
    source=words(text)
    monkeypatch.setattr(pa,'read_crop',lambda *args:words(text,30))
    class Package:
        def image(self,*args):pass
    with pymupdf.open() as doc:
        doc.new_page(width=600,height=400)
        page,report=vp.compile_page({'blocks':[block(invented,bbox=(0,0,1000,1000))]},
            source,SourceDisplay(doc,0),0,Package(),{},pixel_words=source,note_context={'cache':tmp_path/'cache','labels':{},'owners':{},'catalogue':{},'refs':{}})
    record=report['blocks'][0]
    assert record['published_tokens']==text.split()
    assert record['original_model_tokens']==invented.split()
    assert any(d['status']=='UNRESOLVED' and d['chosen']==d['source'] and d['model']
               for d in record['adjudication'])
    assert record['marked']


def test_marker_lexical_view_reserves_only_the_proved_physical_occurrence():
    import copy
    from cps.services.reflow import vision_pipeline as vp
    source=[dict(text='Clear',bbox=[10,20,40,32]),
            dict(text='7',bbox=[41,17,45,23]),
            dict(text='words',bbox=[47,20,77,32]),
            dict(text='7',bbox=[150,20,155,32])]
    before=copy.deepcopy(source)
    tokens=['Clear','<n:first>','words','7']
    proofs={1:dict(status='PROVED',label='7',source_bbox=[41,17,45,23])}
    model,lexical=vp._lexical_marker_view(tokens,source,proofs)
    assert model==tokens and [w['text']for w in lexical]==['Clear','words','7']
    assert lexical[-1]['bbox']==[150,20,155,32]
    assert source==before
    # A genuinely separate printed slot remains a separate reference.
    tokens=['Clear','<n:first>','words','<n:second>']
    proofs[3]=dict(status='PROVED',label='7',source_bbox=[150,20,155,32])
    model,lexical=vp._lexical_marker_view(tokens,source,proofs)
    assert model==tokens and [w['text']for w in lexical]==['Clear','words']


def test_marker_lexical_view_strips_only_an_exact_proved_raised_suffix():
    import copy
    from cps.services.reflow import vision_pipeline as vp
    source=[dict(text='Clear7,',bbox=[10,20,50,32]),
            dict(text='words',bbox=[55,20,85,32]),
            dict(text='Clear7,',bbox=[150,20,190,32])]
    before=copy.deepcopy(source)
    tokens=['Clear7,','<n:first>','words','Clear7,']
    proof=dict(status='PROVED',label='7',source_bbox=[45,17,49,24],
               deduplicated_literal_token=0,literal_display_text='Clear,')
    model,lexical=vp._lexical_marker_view(tokens,source,{1:proof})
    assert model==['Clear,','<n:first>','words','Clear7,']
    assert [w['text']for w in lexical]==['Clear,','words','Clear7,']
    assert lexical[0]['bbox']==source[0]['bbox']
    assert tokens==['Clear7,','<n:first>','words','Clear7,'] and source==before


def test_unbound_or_numeric_role_unproved_markers_are_never_stripped():
    from cps.services.reflow import vision_pipeline as vp
    source=[dict(text='Clear',bbox=[10,20,40,32]),
            dict(text='7',bbox=[41,20,45,32]),dict(text='words',bbox=[47,20,77,32])]
    tokens=['Clear','<n:first>','words']
    for reason in [vp._MARKER_ROLE_UNPROVED,vp._MARKER_SLOT_UNBOUND]:
        proof=dict(status='UNPROVED',source_bbox=[41,20,45,32],reason=reason,
                   anchor_before_bbox=source[0]['bbox'],anchor_after_bbox=source[2]['bbox'])
        assert vp._lexical_marker_view(tokens,source,{1:proof})==(tokens,source)
    # An unbounded entire-block rectangle cannot reserve a source number.
    proof=dict(status='UNPROVED',source_bbox=[0,0,100,100],
               reason=vp._MARKER_GROUP_AMBIGUOUS)
    assert vp._lexical_marker_view(tokens,source,{1:proof})==(tokens,source)


def test_bounded_unknown_marker_identity_remains_opaque_without_claiming_a_label():
    from cps.services.reflow import vision_pipeline as vp
    source=[dict(text='Clear',bbox=[10,20,40,32]),
            dict(text='3*',bbox=[41,17,48,23]),dict(text='words',bbox=[50,20,80,32]),
            dict(text='3',bbox=[150,20,155,32])]
    tokens=['Clear','<n:first>','words','3']
    proof=dict(status='UNPROVED',source_bbox=[41,17,48,23],
               reason=vp._MARKER_GROUP_AMBIGUOUS,
               anchor_before_bbox=source[0]['bbox'],anchor_after_bbox=source[2]['bbox'])
    model,lexical=vp._lexical_marker_view(tokens,source,{1:proof})
    assert model==tokens and [w['text']for w in lexical]==['Clear','words','3']
    assert proof['status']=='UNPROVED' and 'label'not in proof
    # A letter-looking unknown reading is not safe to classify as marker text.
    assert vp._lexical_marker_view(tokens,[dict(source[1],text='B')],{1:proof})[1][0]['text']=='B'


@pytest.mark.parametrize('kind',['bare','suffix','opaque'])
def test_compile_reserves_proved_marker_before_word_conservation(tmp_path,monkeypatch,kind):
    import copy,pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    text='The clear sample remains readable near the ordinary village before sunset today.'
    source=words(text)
    glyph=[58,16,62,24];suffix=kind=='suffix'
    if suffix:source[2]=dict(source[2],text='sample7',bbox=[40,20,63,32])
    else:source.insert(3,dict(text='3*'if kind=='opaque'else'7',bbox=glyph,confidence=99))
    original=copy.deepcopy(source)
    proposed=text.replace('sample','sample7<n:n1>'if suffix else 'sample<n:n1>')
    expected=text.replace('sample','sample <n:n1>').split()
    calls=[];dispatches=[]
    def proof(tokens,optical,display,box,context,pno,literal=None):
        calls.append((context.get('marker_reader'),copy.deepcopy(optical)))
        i=tokens.index('<n:n1>')
        result=(dict(status='UNPROVED',source_bbox=glyph,reason=vp._MARKER_GROUP_AMBIGUOUS,
                     anchor_before_bbox=original[2]['bbox'],anchor_after_bbox=original[4]['bbox'])
                if kind=='opaque'else dict(status='PROVED',label='7',source_bbox=glyph))
        if tokens[i-1]=='sample7':result.update(deduplicated_literal_token=i-1,literal_display_text='sample')
        return {i:result}
    def dispatch(*args):dispatches.append(args);raise AssertionError('No added marker dispatch is allowed')
    monkeypatch.setattr(vp,'_placeholder_display_evidence',proof)
    monkeypatch.setattr(vp,'_literal_note_refs',lambda *args,**kwargs:{})
    monkeypatch.setattr(pa,'read_crop',lambda *args:copy.deepcopy(source))
    context=dict(cache=tmp_path/'cache',labels={'7':[(0,'n1')]},owners={},
                 catalogue={(0,'n1'):{'target':'v4-note-0-n1','label':'99'}},refs={},marker_reader=dispatch)
    class Package:
        def image(self,*args):pass
    with pymupdf.open()as doc:
        doc.new_page(width=600,height=400)
        page,report=vp.compile_page({'blocks':[block(proposed,bbox=(0,0,1000,700)),block('',ident='n1',role='footnote',marker='99',bbox=(0,900,1000,1000))]},
            source,SourceDisplay(doc,0),0,Package(),{},pixel_words=source,note_context=context)
    assert report['blocks'][0]['published_tokens']==expected
    assert source==original and calls[-1][1]==original
    assert len(calls)==2 and calls[0][0]is None and calls[1][0]is dispatch
    assert not dispatches
    if kind=='opaque':
        assert report['note_marker_disclosures'] and report['note_marker_disclosures'][0]['status']=='DEGRADED_SOURCE_CONTEXT'
        assert 'class="source-disclosure-ref"'in page['body'][0]
    else:
        assert 'class="noteref"'in page['body'][0] and '<sup>7</sup>'in page['body'][0]
    assert not any(d.get('source')==['7']for d in report['blocks'][0]['adjudication'])


def test_pixel_audit_does_not_turn_missing_terminal_punctuation_into_a_word_conflict():
    from cps.services.reflow.pixel_adjudication import audit_agreement
    source=words('The clear sample remains readable today.')
    witness=[dict(w,text='today' if w['text']=='today.' else w['text'])for w in source]
    assert audit_agreement([w['text']for w in source],source,witness)=={}
    damaged=[dict(w,text='t0day' if w['text']=='today' else w['text'])for w in witness]
    assert set(audit_agreement([w['text']for w in source],source,damaged))=={5}


@pytest.mark.parametrize('recognized',['t','+'])
def test_unknown_reference_slot_is_not_an_ordinary_lexical_word(recognized):
    from cps.services.reflow import vision_pipeline as vp
    tokens=['Clear','minerals,’‡','<n:note>','and','ordinary','samples.']
    source=[dict(text='minerals,’',bbox=[30,20,75,32]),
            dict(text=recognized,bbox=[77,22,81,30]),dict(text='and',bbox=[84,20,102,32])]
    proof=dict(status='UNPROVED',source_bbox=[77,22,81,30],
        reason=vp._MARKER_SLOT_UNBOUND,
        anchor_before_bbox=source[0]['bbox'],anchor_after_bbox=source[2]['bbox'])
    model,lexical=vp._lexical_marker_view(tokens,source,{2:proof})
    assert model==['Clear','minerals,’','<n:note>','and','ordinary','samples.']
    assert [w['text']for w in lexical]==['minerals,’','and']
    assert source[1]['text']==recognized,'reservation must not edit the original occurrence inventory'
    assert proof['lexical_literal_suffix']=='‡'


def test_native_live_text_without_a_source_occurrence_cannot_skip_pixel_audit(tmp_path):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp
    with pymupdf.open()as doc:
        page=doc.new_page(width=300,height=400)
        page.insert_text((30,60),'A clear sample remains readable.')
        native=dict(pno=0,body=['<p>A clear sample remains readable.</p>'],asides=[])
        records=[dict(id='native-0',role='body',source_regions=[])]
        with pytest.raises(ValueError,match='Native live text has no uniquely bound source occurrence'):
            vp._audit_native_pixels(doc,native,records,tmp_path,object(),{})


def test_agreeing_layer_and_reader_do_not_override_two_own_pixel_readings():
    from cps.services.reflow.pixel_adjudication import adjudicate
    original=words('The clear sample crosses quiet fie1ds before sunset today.')
    pixels=[dict(w,text='fields' if w['text']=='fie1ds' else w['text'])for w in original]
    result=adjudicate([w['text']for w in original],original,pixels,corroboration=[dict(w)for w in pixels])
    assert result['tokens']=='The clear sample crosses quiet fields before sunset today.'.split()
    assert 5 in result['marked']
    assert result['word_notes'][5]['source']==['fie1ds'] and result['word_notes'][5]['chosen']==['fields']
    assert result['word_notes'][5]['status']=='RESOLVED'
    remote=[dict(w,bbox=[w['bbox'][0],200,w['bbox'][2],212])for w in pixels]
    uncertain=adjudicate([w['text']for w in original],original,pixels,corroboration=remote)
    assert uncertain['tokens']==[w['text']for w in original]


@pytest.mark.parametrize('punctuation',[';', '.'])
def test_split_terminal_punctuation_does_not_create_a_word_or_duplicate_symbol(punctuation):
    source=words('The clear sample remains readable today '+punctuation)
    proposed=('The clear sample remains readable today'+punctuation).split()
    result=adjudicate(proposed,source,[dict(w)for w in source])
    assert result['tokens']==proposed
    assert result['marked']==set(),'terminal punctuation segmentation is not a word correction'
    damaged_source=[dict(w,text='t0day' if w['text']=='today' else w['text'])for w in source]
    damaged=adjudicate(proposed,damaged_source,[dict(w)for w in damaged_source])
    assert damaged['tokens'][-1].endswith(punctuation)
    assert 5 in damaged['marked'],'letter disagreement still needs its own uncertainty'


@pytest.mark.parametrize('notice_kind',['punctuation','source_context','source_raster'])
def test_generated_native_notice_is_not_an_unbound_authored_paragraph(tmp_path,monkeypatch,notice_kind):
    import pymupdf
    from cps.services.reflow import assemble,build_epub,vision_pipeline as vp,pixel_adjudication as pa
    text='The clear sample remains readable beside the quiet window.'
    with pymupdf.open()as doc:
        original=doc.new_page(width=500,height=500)
        original.insert_text((30,70),text,fontsize=10)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in original.get_text('words')]
        element=assemble.Element('p',runs=[['t',text]],pno=0,bbox=[30,58,450,73])
        book=assemble.Book(elements=[element],pages={0:[element]})
        notice=(build_epub._punctuation_notice(0,0)if notice_kind=='punctuation'else
                build_epub._source_check_notice('Original source retained','original-p0000.xhtml#page','View original source'))
        if notice_kind=='source_raster':
            notice='<p><span class="source-raster"><img src="images/source.png" alt="Original source lettering"/></span></p>'
        native=dict(pno=0,body=['<p>'+text+'</p>',notice],asides=[])
        records=vp._native_seam_blocks(doc,book,0,native)
        assert records[0]['role']=='body' and records[0]['source_regions']
        assert records[1]['role']=='other' and records[1]['source_regions']==[]
        calls=[]
        monkeypatch.setattr(pa,'read_crop',lambda *args:(calls.append(args[1])or source))
        vp._audit_native_pixels(doc,native,records,tmp_path,object(),{})
        assert len(calls)==1 and native['body']==['<p>'+text+'</p>',notice]
        # Equal prose without the generated notice class remains subject to
        # normal physical ownership, even when its words describe a notice.
        unbound=dict(pno=0,body=['<p>Original source retained</p>'],asides=[])
        authored=vp._native_seam_blocks(doc,book,0,unbound)
        assert authored[0]['role']=='body'
        with pytest.raises(ValueError,match='Native live text has no uniquely bound source occurrence'):
            vp._audit_native_pixels(doc,unbound,authored,tmp_path,object(),{})


def test_partial_printed_hyphen_atom_cannot_certify_truncating_a_complete_word():
    source=words('Please inspect the printed wor-')
    proposed='Please inspect the printed workpieces:'.split()
    result=adjudicate(proposed,source,[dict(w)for w in source])
    assert result['tokens']==proposed
    assert result['word_notes'][4]['status']=='UNRESOLVED'
    assert not any(d['status']=='RESOLVED'for d in result['decisions'])

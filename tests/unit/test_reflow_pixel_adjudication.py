"""Synthetic transcription alternatives and occurrence geometry written here."""
import pytest
from cps.services.reflow.pixel_adjudication import adjudicate

pytestmark=pytest.mark.unit


def words(text, confidence=99):
    return [dict(text=t,bbox=[i*20,20,i*20+18,32],confidence=confidence)
            for i,t in enumerate(text.split())]


@pytest.mark.parametrize('conflict', [False, True])
def test_positive_own_optical_word_does_not_need_a_missing_neighbor_read(conflict):
    source=words('The clear sample stays readable.')
    witness=words('The clcar sample stays readable.') if conflict else []
    result=adjudicate('The clear sample stays readable.'.split(),source,witness,uncertain={1})
    if conflict:
        from cps.services.reflow.pixel_adjudication import audit_agreement
        assert 1 in audit_agreement(result['tokens'],source,witness)
    else:
        assert result['tokens']==[w['text']for w in source]
        assert result['marked']==set()


@pytest.mark.parametrize('settled', [False, True])
def test_valid_printed_ordinal_is_not_certified_as_opaque_ocr_letters(settled):
    from cps.services.reflow.pixel_adjudication import CropWitness
    source=words('Read the Bth: sample now.')
    witness=CropWitness(words('Read the Bth: sample now.'))
    witness.word_readings=lambda word:([words('Read the 8th: sample now.')]*2 if settled else [])
    result=adjudicate('Read the 8th: sample now.'.split(),source,witness)
    assert result['tokens'][2]=='8th:'
    assert result['word_notes'][2]['status']==('RESOLVED' if settled else 'UNRESOLVED')
    assert 2 in result['marked']
    assert bool(result['word_notes'][2].get('source_image_only')) is not settled


def test_conflicting_valid_printed_ordinal_still_wins_when_pixels_prove_it():
    source=words('Read the 9th sample now.')
    result=adjudicate('Read the 8th sample now.'.split(),source,source)
    assert result['tokens'][2]=='9th'
    assert result['word_notes'][2]['status']=='RESOLVED'
    assert 2 in result['marked']


def test_complete_two_column_territory_keeps_column_order(monkeypatch):
    from types import SimpleNamespace
    from cps.services.reflow import vision_pipeline as pipeline
    import pymupdf
    source=[]
    for x,prefix in [(10,'Left'),(150,'Right')]:
        for y in [20,40,60]:
            source.extend([dict(text=prefix,bbox=[x,y,x+35,y+10]),
                           dict(text=str(y),bbox=[x+40,y,x+55,y+10])])
    display=SimpleNamespace(rect=pymupdf.Rect(0,0,250,120))
    block=dict(id='sample',role='body',text='Synthetic columns',bbox=[0,0,1,1])
    result=pipeline._owned_block_words(block,dict(blocks=[block]),source,display,
        regions={'sample':[[0,0,250,120]]},ordered=True)
    assert [w['text']for w in result]==[w['text']for w in source]


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


@pytest.mark.parametrize('source_confidence',[30,99])
def test_agreeing_words_do_not_erase_unresolved_reader_uncertainty(source_confidence):
    text='The clear sample remains readable today.'
    result=adjudicate(text.split(),words(text,source_confidence),words(text,30),uncertain={3})
    assert result['marked']==({3}if source_confidence<90 else set())
    if source_confidence<90:assert result['word_notes'][3]['status']=='UNRESOLVED'


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
    # Exercise the damaged-layer audit independently of visible-font proof.
    from cps.services.reflow import native_text
    monkeypatch.setattr(native_text,'drawn_word_authority',lambda *args:False)
    monkeypatch.setattr(native_text,'drawn_word_authorities',lambda page,words:[False]*len(words))
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
    # Exercise the damaged-layer audit independently of visible-font proof.
    from cps.services.reflow import native_text
    monkeypatch.setattr(native_text,'drawn_word_authority',lambda *args:False)
    monkeypatch.setattr(native_text,'drawn_word_authorities',lambda page,words:[False]*len(words))
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
    # Exercise the damaged-layer audit independently of visible-font proof.
    from cps.services.reflow import native_text
    monkeypatch.setattr(native_text,'drawn_word_authority',lambda *args:False)
    monkeypatch.setattr(native_text,'drawn_word_authorities',lambda page,words:[False]*len(words))
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
    # Exercise the damaged-layer audit independently of visible-font proof.
    from cps.services.reflow import native_text
    monkeypatch.setattr(native_text,'drawn_word_authority',lambda *args:False)
    monkeypatch.setattr(native_text,'drawn_word_authorities',lambda page,words:[False]*len(words))
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
    # Exercise the damaged-layer audit independently of visible-font proof.
    from cps.services.reflow import native_text
    monkeypatch.setattr(native_text,'drawn_word_authority',lambda *args:False)
    monkeypatch.setattr(native_text,'drawn_word_authorities',lambda page,words:[False]*len(words))
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


def test_printed_separate_words_do_not_fuse_across_a_row_change():
    source=words('Please read clear bridge beside the quiet window today.')
    source[3]['bbox']=[0,40,18,52]
    result=adjudicate('Please read clearbridge beside the quiet window today.'.split(),source,source)
    assert result['tokens'][2:4]==['clear','bridge']
    assert result['tokens'][4]=='beside'
    assert result['decisions'][0]['status']=='RESOLVED'


def test_local_two_pixel_views_cannot_certify_accent_absence():
    from cps.services.reflow.pixel_adjudication import refine_fallback
    source=words('Please read café beside the quiet window today.')
    witness=[dict(w,text='cafe'if w['text']=='café'else w['text'])for w in source]
    result=refine_fallback([w['text']for w in source],witness,{'read','cafe'},source=source,corroboration=witness)
    assert result['tokens'][2]=='café'
    assert result['word_notes'][2]['status']=='UNRESOLVED'


def test_native_footnote_receives_its_own_independent_pixel_audit(tmp_path,monkeypatch):
    # Exercise the damaged-layer audit independently of visible-font proof.
    from cps.services.reflow import native_text
    monkeypatch.setattr(native_text,'drawn_word_authority',lambda *args:False)
    monkeypatch.setattr(native_text,'drawn_word_authorities',lambda page,words:[False]*len(words))
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


def test_unplaced_terminal_symbol_is_not_removed_then_reintroduced_as_a_word():
    import copy
    from cps.services.reflow import vision_pipeline as vp
    tokens=['Clear','specimens*','<n:note>']
    source=[dict(text='Clear',bbox=[10,20,35,32]),
            dict(text='specimens*',bbox=[38,20,95,32])]
    original=copy.deepcopy(source)
    proof=dict(status='UNPROVED',source_bbox=[38,20,98,32],
        reason=vp._MARKER_GROUP_INCOMPLETE,
        anchor_before_bbox=source[-1]['bbox'],anchor_after_bbox=None)
    model,lexical=vp._lexical_marker_view(tokens,source,{2:proof})
    assert model==tokens
    assert lexical==source==original
    assert 'lexical_literal_suffix' not in proof
    assert 'lexical_placeholder' not in proof


@pytest.mark.parametrize('kind',['plain_pixels','attached_pixels','crop_omits_symbol'])
def test_unplaced_literal_survives_lexical_adjudication_exactly_once(tmp_path,monkeypatch,kind):
    import copy,pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    text='The clear sample remains readable near the ordinary village before sunset today.'
    source=words(text)
    if kind!='plain_pixels':source[2]=dict(source[2],text='sample*')
    witness=copy.deepcopy(source)
    if kind=='crop_omits_symbol':witness[2]['text']='sample'
    monkeypatch.setattr(pa,'read_crop',lambda *args:copy.deepcopy(witness))
    monkeypatch.setattr(vp,'_literal_note_refs',lambda *args,**kwargs:{})
    def proof(tokens,*args,**kwargs):
        return {tokens.index('<n:n1>'):dict(status='UNPROVED',source_bbox=[10,10,300,40],
            reason=vp._MARKER_SLOT_UNBOUND)}
    monkeypatch.setattr(vp,'_placeholder_display_evidence',proof)
    context=dict(cache=tmp_path/'cache',labels={'*':[(0,'n1')]},owners={},
        catalogue={(0,'n1'):{'target':'v4-note-0-n1','label':'*'}},refs={})
    class Package:
        def image(self,*args):pass
    with pymupdf.open()as doc:
        doc.new_page(width=600,height=400)
        _,report=vp.compile_page({'blocks':[block(text.replace('sample','sample*<n:n1>'),bbox=(0,0,1000,700)),
            block('',ident='n1',role='footnote',marker='*',bbox=(0,900,1000,1000))]},
            source,SourceDisplay(doc,0),0,Package(),{},pixel_words=source,note_context=context)
    record=report['blocks'][0]
    assert record['published_tokens']==text.replace('sample','sample* <n:n1>').split()
    assert not record['marked'],'only glyph identity is uncertain; the unchanged lexical word is not a correction'
    assert not record['adjudication']
    assert report['note_marker_disclosures'],'the unproved reference still requires its own source disclosure'


def test_agreeing_word_pixel_audit_keeps_indices_across_typed_references():
    from cps.services.reflow.pixel_adjudication import audit_agreement
    source=words('Clear samples remain readable.')
    witness=[dict(w,text='samp1es' if w['text']=='samples' else w['text'])for w in source]
    notes=audit_agreement(['Clear','<n:note>','samples','remain','readable.'],source,witness)
    assert set(notes)=={2}
    assert notes[2]['source']==notes[2]['model']==notes[2]['chosen']==['samples']
    assert notes[2]['alternative']==['samp1es']
    assert notes[2]['status']=='UNRESOLVED'


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
    # Exercise the damaged-layer audit independently of visible-font proof.
    from cps.services.reflow import native_text
    monkeypatch.setattr(native_text,'drawn_word_authority',lambda *args:False)
    monkeypatch.setattr(native_text,'drawn_word_authorities',lambda page,words:[False]*len(words))
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


def test_unavailable_secondary_engine_returns_explicit_failure_without_dispatch(tmp_path,monkeypatch):
    from cps.services.reflow import pixel_adjudication as pa
    import shutil,subprocess
    monkeypatch.setattr(shutil,'which',lambda name:str(tmp_path/'missing-recognizer'))
    def forbidden(*args,**kwargs):
        raise AssertionError('Unavailable executable must not be dispatched')
    monkeypatch.setattr(subprocess,'run',forbidden)
    result=pa.read_crop(None,[[0,0,10,10]],tmp_path)
    assert result==[]
    assert result.failure=='RuntimeError: Independent OCR executable is unavailable'


def test_projection_keeps_generated_control_tails_at_their_original_siblings():
    from tests.unit.test_round4_defects import plain
    from xml.etree import ElementTree as ET
    node=ET.fromstring('<p xmlns="http://www.w3.org/1999/xhtml"><span>as<a>†</a> a</span><a>†</a> <span>quiet harbor.</span></p>')
    assert plain(node)=='as a quiet harbor.'


def test_rejected_first_read_cannot_remove_original_terminal_punctuation():
    original=words('The quiet river.')
    proposed='The quiet river'.split()
    result=adjudicate(proposed,original,words('The quiet river'),unresolved_source=True)
    assert result['tokens']==['The','quiet','river.']
    assert result['word_notes'][2]['status']=='UNRESOLVED'
    assert result['word_notes'][2]['model']==['river']


def test_fused_recognition_needs_pixel_space_proof_to_clear_unchanged_words():
    from cps.services.reflow.pixel_adjudication import audit_agreement
    original=words('A distinct sample remains readable.')
    merged=dict(original[0],text='Adistinct',bbox=[10,20,25,30])
    witness=[merged,*original[2:]]
    tokens=[w['text']for w in original]
    assert set(audit_agreement(tokens,original,witness))=={0,1}
    observed=[]
    def blank(a,b):
        observed.append((a['text'],b['text']));return True
    assert audit_agreement(tokens,original,witness,boundary_proof=blank)=={}
    assert observed==[('A','distinct')]
    assert set(audit_agreement(tokens,original,witness,boundary_proof=lambda *args:False))=={0,1}


def test_separate_printed_words_keep_their_boundary_across_a_row_change():
    source=words('We read (cafe bleu beside the window.')
    for w in source[3:]:w['bbox']=[w['bbox'][0]-60,w['bbox'][1]+18,w['bbox'][2]-60,w['bbox'][3]+18]
    result=adjudicate('We read (cafébleu beside the window.'.split(),source,source)
    assert result['tokens']==['We','read','(café','bleu','beside','the','window.']
    assert result['decisions'][0]['method']=='pixel_word_boundary_only'
    assert result['decisions'][0]['status']=='UNRESOLVED'
    assert result['marked']=={2,3},'the accent still needs its own pixel proof'


@pytest.mark.parametrize('word',['<em>remalns</em>','re<em>malns</em>'])
def test_native_pixel_correction_preserves_inline_word_formatting(tmp_path,monkeypatch,word):
    # Exercise the damaged-layer audit independently of visible-font proof.
    from cps.services.reflow import native_text
    monkeypatch.setattr(native_text,'drawn_word_authority',lambda *args:False)
    monkeypatch.setattr(native_text,'drawn_word_authorities',lambda page,words:[False]*len(words))
    import re
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa,vision_pixels
    fragment='<p>The clear sample '+word+' readable today.</p>'
    class Package:
        def disclosure_image(self,*args):pass
    with pymupdf.open()as doc:
        leaf=doc.new_page(width=400,height=500)
        leaf.insert_text((30,70),'The clear sample remalns readable today.',fontsize=10)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in leaf.get_text('words')]
        witness=[dict(w,text='remains'if w['text']=='remalns'else w['text'])for w in source]
        monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
        monkeypatch.setattr(vision_pixels,'words',lambda *args:witness)
        page=dict(pno=0,body=[fragment],asides=[])
        records=[dict(id='native-0',role='body',source_regions=[[25,55,390,80]])]
        vp._audit_native_pixels(doc,page,records,tmp_path,Package(),{})
    authored=re.sub(r'<a[^>]*class="source-disclosure-ref"[^>]*>.*?</a>','',page['body'][0])
    assert vp._seam_text(authored)=='The clear sample remains readable today.'
    assert '<em>'in authored and '</em>'in authored
    assert records[0]['marked']==[3]
    assert any(d['status']=='RESOLVED'and d['chosen']==['remains']for d in records[0]['adjudication'])


def test_alternate_optical_alphabet_settles_a_macron_without_erasing_its_mark():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Please read the clear sampel beside the window.')
    model='Please read the clear samplē beside the window.'.split()
    witness=pa.CropWitness(words('Please read the clear sample beside the window.'))
    witness.alternates=lambda:[words('Please read the clear samplē beside the window.',96),
                               words('Please read the clear samplē beside the window.',95)]
    result=adjudicate(model,source,witness)
    assert result['tokens']==model
    assert result['word_notes'][4]['status']=='RESOLVED'
    assert result['word_notes'][4]['method']=='two_alternate_alphabet_pixel_readings'
    assert result['marked']=={4}


def test_alternate_alphabet_does_not_claim_absence_of_an_unrepresented_mark():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Please read the clear sample beside the window.')
    witness=pa.CropWitness(source)
    witness.alternates=lambda:[source,source]
    result=adjudicate('Please read the clear samplẹ beside the window.'.split(),source,witness)
    assert result['word_notes'][4]['status']=='UNRESOLVED'
    assert result['tokens'][4]=='samplẹ'


def test_two_alternate_pixel_reads_restore_the_native_number_at_its_own_occurrence():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Please read section 4.2.17; before opening the box.')
    witness=pa.CropWitness(words('Please read section 4.2-17; before opening the box.',40))
    witness.alternates=lambda:[source,source]
    result=adjudicate('Please read section 4.2-17; before opening the box.'.split(),source,witness)
    assert result['tokens']==[w['text']for w in source]
    assert result['word_notes'][3]['status']=='RESOLVED'


def test_alternate_optical_data_must_match_the_pinned_network(tmp_path):
    from cps.services.reflow import pixel_adjudication as pa
    (tmp_path/'LatinRaw.traineddata').write_bytes(b'not the optical model')
    assert pa._verified_latin_directory((str(tmp_path),))is None


def test_settled_edit_does_not_run_the_optional_recognizer():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Please read the clear sample beside the window.')
    witness=pa.CropWitness(source)
    def unwanted():raise AssertionError('an already proved edit needs no extra recognition')
    witness.alternates=unwanted
    result=adjudicate('Please read the clear sampel beside the window.'.split(),source,witness)
    assert result['tokens']==[w['text']for w in source]
    assert result['word_notes'][4]['status']=='RESOLVED'


def test_unsettled_alternate_word_publishes_its_own_ink_not_wrong_live_text(tmp_path,monkeypatch):
    from tests.unit import test_round4_defects as ink
    from cps.services.reflow import pixel_adjudication as pa
    page=ink.InkPage();printed='The clear sample remains readable beside the open window.'
    page.text(printed,'body',model=printed.replace('remains','remalns'))
    source=[dict(text=w[4],bbox=list(w[:4]),confidence=40)for w in page.page.get_text('words')]
    def crop(display,regions,cache):
        # The first witness and the proposed reading agree on the bad spelling;
        # two alternate pixel views disagree but cannot certify a replacement.
        bad=pa.CropWitness([dict(w,text='remalns'if w['text']=='remains'else w['text'])for w in source])
        bad.alternates=lambda:[source,source]
        return bad
    monkeypatch.setattr(pa,'read_crop',crop)
    _,report,docs=ink.convert_ink(tmp_path,monkeypatch,[page])
    text=' '.join(ink.body_texts(docs))
    assert 'remalns'not in text
    images=[n for root in docs.values()for n in root.iter()if n.get('class')=='source-word-image']
    assert len(images)==1
    assert len(images[0])==1 and images[0][0].tag.endswith('img')
    assert 'The clear sample'in text and 'readable beside the open window.'in text
    assert report['pages'][0]['blocks'][0]['source_image_words']==[3]


@pytest.mark.parametrize('dot',[False,True])
def test_underdot_absence_requires_complete_blank_pixels_below_a_proved_baseline(dot):
    import pymupdf
    from cps.services.reflow import pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    with pymupdf.open()as doc:
        page=doc.new_page(width=200,height=120)
        page.insert_text((30,50),'calm',fontsize=14)
        word=page.get_text('words')[0]
        if dot:page.draw_circle((46,52.5),.7,color=(0,0,0),fill=(0,0,0))
        assert pa.below_dot_absent(SourceDisplay(doc,0),dict(text='calm',bbox=list(word[:4])))is(not dot)


@pytest.mark.parametrize('dot',[False,True])
def test_alternate_alphabet_needs_pixels_to_remove_a_below_baseline_mark(dot):
    import pymupdf
    from cps.services.reflow import pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    with pymupdf.open()as doc:
        page=doc.new_page(width=350,height=120)
        page.insert_text((30,50),'Please read calm beside the window.',fontsize=14)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        if dot:page.draw_circle((source[2]['bbox'][0]+8,52.5),.7,color=(0,0,0),fill=(0,0,0))
        witness=pa.CropWitness(source);witness.alternates=lambda:[source,source]
        witness.mark_absence=lambda word:pa.below_dot_absent(SourceDisplay(doc,0),word)
        result=adjudicate('Please read cạlm beside the window.'.split(),source,witness)
    assert result['tokens'][2]==('cạlm'if dot else'calm')
    assert result['word_notes'][2]['status']==('UNRESOLVED'if dot else'RESOLVED')


def test_unsettled_views_cannot_force_a_wrong_live_spelling_or_choose_each_other():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('clear sampel beside')
    witness=pa.CropWitness(source)
    witness.alternates=lambda:[words('clear sample beside',40),words('clear samble beside',35)]
    result=dict(tokens='clear sampel beside'.split(),marked={1},word_notes={})
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness)
    assert result['tokens'][1]=='sampel','no alternate guess gains authority'
    assert result['word_notes'][1]['source_image_only']
    assert result['word_notes'][1]['status']=='UNRESOLVED'


def test_installed_alternate_network_reads_printed_macron_pixels(tmp_path):
    import pymupdf,shutil
    from cps.services.reflow import pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    engine=shutil.which('tesseract')
    if not engine or not pa._latin_directory(engine):pytest.skip('pinned alternate optical data not installed')
    with pymupdf.open()as doc:
        page=doc.new_page(width=400,height=160)
        font=pymupdf.Font('tiro')
        writer=pymupdf.TextWriter(page.rect)
        writer.append((25,65),'Please read māple beside the clear window.',font=font,fontsize=14)
        writer.write_text(page)
        witness=pa.read_crop(SourceDisplay(doc,0),[[20,45,380,75]],tmp_path)
        views=witness.alternates()
    assert len(views)==2
    assert all('māple'in [w['text']for w in view]for view in views)


def test_alternate_recognition_joins_only_consecutive_rows_in_the_same_column():
    from cps.services.reflow import pixel_adjudication as pa
    assert pa.alternate_regions([[20,20,180,32],[30,35,175,47],[30,50,160,62]])==[[20,20,180,62]]
    assert pa.alternate_regions([[20,20,100,32],[130,35,200,47]])==[[20,20,100,32],[130,35,200,47]]
    assert pa.alternate_regions([[20,20,180,32],[20,100,180,112]])==[[20,20,180,32],[20,100,180,112]]


def test_refinement_retains_an_unresolved_original_word_footprint():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Please read the sampel beside the quiet window.')
    result=pa.refine_fallback([w['text']for w in source],
        words('Please read the sample beside the quiet window.'),{'unrelated'},source=source)
    assert result['tokens'][3]=='sampel'
    assert result['word_notes'][3]['status']=='UNRESOLVED'
    assert result['boxes'][3]==source[3]['bbox']


def test_a_punctuation_atom_cannot_authorize_deleting_a_word_prefix():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('read , beside')
    source[1]['bbox']=[35,20,38,32]
    view=words('read scope beside')
    witness=pa.CropWitness([dict(w,confidence=40)for w in source]);witness.alternates=lambda:[view,view]
    result=adjudicate('read microscope beside'.split(),source,witness)
    assert result['tokens'][1]=='microscope'
    assert result['word_notes'][1]['status']=='UNRESOLVED'


def test_exact_alternate_pixels_settle_a_spurious_agreeing_reading_conflict():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Please read every original detail beside the window.')
    witness=pa.CropWitness([dict(w,text='eveiy'if w['text']=='every'else w['text'])for w in source])
    witness.alternates=lambda:[source,source]
    conflicts=pa.audit_agreement([w['text']for w in source],source,witness)
    assert set(conflicts)=={2}
    result=dict(tokens=[w['text']for w in source],marked={2},word_notes=conflicts)
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness)
    assert result['marked']==set()
    assert result['word_notes']=={}
    assert result['verified_unchanged']==[2]


def test_one_coherent_frame_pair_can_settle_a_number_but_conflicting_pairs_cannot():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Please read section 4.2.17; before opening the box.')
    bad=words('Please read section 4.2-17; before opening the box.',40)
    witness=pa.CropWitness(bad)
    witness.alternates=lambda:[source,source,bad,bad]
    result=adjudicate([w['text']for w in bad],source,witness)
    assert result['tokens']==[w['text']for w in source]
    assert result['word_notes'][3]['status']=='RESOLVED'
    conflict=words('Please read section 4.2-17; before opening the box.')
    witness.alternates=lambda:[source,source,conflict,conflict]
    result=adjudicate([w['text']for w in bad],source,witness)
    assert result['word_notes'][3]['status']=='UNRESOLVED'


def test_an_owned_english_word_witness_prevents_spurious_image_replacement():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('The careful sample remains beside the window.')
    witness=pa.CropWitness(source)
    witness.alternates=lambda:[words('The careful sampie remains beside the window.',40)]*2
    result=dict(tokens=[w['text']for w in source],marked={2},word_notes={2:dict(source=['sample'],model=['sample'],chosen=['sample'],status='UNRESOLVED')})
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness,allow_clear=False)
    assert not result['word_notes'][2].get('source_image_only')
    assert result['marked']=={2},'A local optical claim remains visibly unverified'


def test_case_only_alternate_noise_keeps_the_owned_literal_word_live():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('The clear SAMPLE remains beside the window.',40)
    witness=pa.CropWitness(words('The clear sampie remains beside the window.',40))
    witness.alternates=lambda:[words('The clear Sample remains beside the window.',40)]*2
    result=dict(tokens=[w['text']for w in source],marked={2},word_notes={2:dict(source=['SAMPLE'],model=['SAMPLE'],chosen=['SAMPLE'],status='UNRESOLVED')})
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness)
    assert not result['word_notes'][2].get('source_image_only')
    assert result['marked']=={2}


def test_partial_printed_wrap_is_not_counterevidence_to_a_complete_word():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('across the mea- while waiting',40)
    witness=pa.CropWitness(source);witness.alternates=lambda:[words('across the mea- while waiting')]*2
    result=dict(tokens='across the meadow while waiting'.split(),marked={2},word_notes={2:dict(source=['mea-'],model=['meadow'],chosen=['meadow'],status='UNRESOLVED')})
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness)
    assert not result['word_notes'][2].get('source_image_only')
    assert result['tokens'][2]=='meadow'


def test_marker_suffix_noise_cannot_replace_its_unchanged_lexical_prefix():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('The clear crystal(2048), remains visible.',40)
    witness=pa.CropWitness(words('The clear crystal(**), remains visible.',40))
    witness.alternates=lambda:[words('The clear crystal(°°), remains visible.',40)]*2
    result=dict(tokens=[w['text']for w in source],marked={2},word_notes={2:dict(source=['crystal(2048),'],model=['crystal(2048),'],chosen=['crystal(2048),'],status='UNRESOLVED')})
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness)
    assert not result['word_notes'][2].get('source_image_only')
    assert result['tokens'][2]=='crystal(2048),'


def test_unplaced_edit_cannot_guess_one_word_image_from_an_alignment_box():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Historical examples remain beside the window.',40)
    witness=pa.CropWitness(words('Histrical examples remain beside the window.',40))
    witness.alternates=lambda:[words('Histoical examples remain beside the window.',40)]*2
    result=dict(tokens=[w['text']for w in source],marked={0},word_notes={0:dict(source=[],model=[],chosen=['Historical'],status='UNRESOLVED',mapping='unplaced_within_span')})
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness)
    assert not result['word_notes'][0].get('source_image_only')
    assert result['word_notes'][0]['mapping']=='unplaced_within_span'


def test_a_printed_wrap_keeps_its_literal_hyphen_after_lexical_pixel_agreement():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('The old bridge‐',40)
    witness=pa.CropWitness(words('The old bridge',0))
    witness.alternates=lambda:[words('The old bridgeE',40)]*2
    result=dict(tokens=[w['text']for w in source],marked={2},word_notes={2:dict(source=['damaged-source-token'],model=['bridge‐'],chosen=['bridge‐'],status='UNRESOLVED')})
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness)
    assert not result['word_notes'][2].get('source_image_only')
    assert result['tokens'][2]=='bridge‐'


@pytest.mark.parametrize('confidence',[40,99])
def test_alternate_noise_cannot_override_a_confident_original_occurrence(confidence):
    from cps.services.reflow import pixel_adjudication as pa
    source=words('A printed note remains beside the window.',confidence)
    witness=pa.CropWitness(words('A printed nate remains beside the window.'))
    witness.alternates=lambda:[words('A printed nate remains beside the window.',40)]*2
    witness.source_occurrences=source
    result=dict(tokens=[w['text']for w in source],marked={2},word_notes={2:dict(source=['note'],model=['note'],chosen=['note'],status='UNRESOLVED')})
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness)
    assert bool(result['word_notes'][2].get('source_image_only'))is(confidence<85)
    assert result['marked']=={2},'Conflicting local reads remain unresolved'


def test_unsettled_identifier_keeps_its_literal_instance_and_source_control():
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Please inspect Item12 beside the window.',40)
    witness=pa.CropWitness(words('Please inspect lteml2 beside the window.',40))
    witness.alternates=lambda:[words('Please inspect Iteml2 beside the window.',40)]*2
    result=dict(tokens=[w['text']for w in source],marked={2},word_notes={2:dict(source=['Item12'],model=['Item12'],chosen=['Item12'],status='UNRESOLVED')})
    pa.disclose_alternate_conflicts(result,{i:w['bbox']for i,w in enumerate(source)},witness)
    assert not result['word_notes'][2].get('source_image_only')
    assert result['tokens'][2]=='Item12' and result['marked']=={2}


def test_unavailable_alternate_data_does_not_widen_the_primary_reading_failure(monkeypatch,tmp_path):
    from cps.services.reflow import pixel_adjudication as pa
    primary=words('The independently read sample remains beside the window.')
    monkeypatch.setattr(pa,'_read_crop',lambda *a,**k:primary)
    def unavailable(engine):raise OSError('synthetic unavailable optical data')
    monkeypatch.setattr(pa,'_latin_directory',unavailable)
    import shutil
    monkeypatch.setattr(shutil,'which',lambda name:'/synthetic/tesseract')
    witness=pa.read_crop(None,[[20,20,200,32]],tmp_path)
    assert witness.alternates()==[]
    assert list(witness)==primary


@pytest.mark.parametrize('displaced',[False,True])
def test_unsettled_extra_pixel_ink_is_not_lost_between_fallback_words(displaced):
    """Authored occurrence lists put one ambiguous upright stroke in the gap."""
    from cps.services.reflow import pixel_adjudication as pa
    source=words('Workers carried a tray outside today.')
    glyph=dict(text='|',confidence=55,bbox=[18.5,22,19.5,31])
    if displaced:glyph['bbox']=[118.5,62,119.5,71]
    witness=pa.CropWitness([source[0],glyph,*source[1:]])
    witness.alternates=lambda:[source,source]
    result=pa.refine_fallback([w['text']for w in source],witness,
        {'workers','carried','a','tray','outside','today'},source=source)
    assert result['tokens']==[w['text']for w in source], 'An ambiguous stroke cannot become a guessed live letter'
    notes=result['word_notes']
    if displaced:
        assert not any(n.get('source_image_only')for n in notes.values())
    else:
        assert notes[1]['source_image_only'], 'Unsettled additional ink must remain in the reading'
        assert notes[1]['source_bbox']==[18.5,20,38,32]
        assert notes[1]['status']=='UNRESOLVED'
        assert 1 in result['marked']
        pa.disclose_alternate_conflicts(result,result['boxes'],witness)
        assert notes[1]['source_bbox']==[18.5,20,38,32], 'An alternate supporting the adjacent word cannot erase the uncertain gap'


def test_fallback_extra_ink_survives_the_real_publication_as_its_own_image(tmp_path,monkeypatch):
    """A generated scan prints a narrow letter that both initial inputs omit."""
    from tests.unit import test_round4_defects as ink
    from cps.services.reflow import pixel_adjudication as pa
    page=ink.InkPage();printed='Before lunch I carried a tray outside today.'
    page.text(printed,'body',model=printed.replace(' I ',' ')+
        ' Green baskets contain ripe apples while purple carts carry fresh bread beside a clean window.')
    original=page.finish
    source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.page.get_text('words')]
    def missing_letter():
        image,optical,answer=original()
        return image,[w for w in optical if w['text']!='I'],answer
    page.finish=missing_letter
    witness=pa.CropWitness([dict(w,text='|',confidence=55)if w['text']=='I'else w for w in source])
    monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
    path,report,docs=ink.convert_ink(tmp_path,monkeypatch,[page])
    record=next(b for b in report['pages'][0]['blocks']if b['id']=='body')
    assert record['fallback']
    assert record['source_image_words']==[2], 'The ambiguous extra letter and its next word must survive as ink'
    text=' '.join(ink.body_texts(docs))
    assert 'Before lunch'in text and 'a tray outside today.'in text
    assert 'carried'not in text, 'The image contains that word once; do not repeat it as live text'
    images=[n for root in docs.values()for n in root.iter()if n.get('class')=='source-word-image']
    assert len(images)==1 and len(images[0])==1
    decision=next(d for d in record['adjudication']if d.get('method')=='unsettled_additional_pixel_ink')
    box=decision['source_bbox']
    for word in source[2:4]:
        x=(word['bbox'][0]+word['bbox'][2])/2;y=(word['bbox'][1]+word['bbox'][3])/2
        assert box[0]<=x<=box[2]and box[1]<=y<=box[3], 'The image owns the omitted stroke and following word'


def test_native_pixel_abort_discloses_only_its_unit_and_keeps_siblings_and_later_pages(tmp_path,monkeypatch):
    # Exercise the damaged-layer audit independently of visible-font proof.
    from cps.services.reflow import native_text
    monkeypatch.setattr(native_text,'drawn_word_authority',lambda *args:False)
    monkeypatch.setattr(native_text,'drawn_word_authorities',lambda page,words:[False]*len(words))
    import pymupdf,zipfile
    from types import SimpleNamespace
    from xml.etree import ElementTree as E
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from tests.unit.test_round4_defects import plain
    first='Quiet readers compare the first printed sample beside the window.'
    sibling='Careful workers keep the second printed sample beside the doorway.'
    later='The next page retains its complete independently printed sentence.'
    failures=[]
    with pymupdf.open()as doc:
        page=doc.new_page(width=430,height=500)
        page.insert_text((30,70),first,fontsize=10)
        page.insert_text((30,180),sibling,fontsize=10)
        page=doc.new_page(width=430,height=500)
        page.insert_text((30,70),later,fontsize=10)
        physical={n:[dict(text=w[4],bbox=list(w[:4]),confidence=99)
                     for w in doc[n].get_text('words')]for n in range(2)}
        def read(display,regions,cache):
            if display.pno==0 and max(box[3]for box in regions)<120:
                failures.append(list(regions))
                raise ValueError('Synthetic native crop recognition failure')
            return physical[display.pno]
        monkeypatch.setattr(pa,'read_crop',read)
        output=tmp_path/'native-scope.epub'
        report=vp.convert(doc,str(output),SimpleNamespace(cache=tmp_path/'cache/readings',key=None))
    with zipfile.ZipFile(output)as archive:
        text=' '.join(plain(E.fromstring(archive.read(n)))for n in archive.namelist()if n.endswith('.xhtml'))
    assert sibling in text and later in text, 'A failed native unit must not consume healthy printed siblings'
    assert first not in text, 'The failed unit must publish its own source image rather than unaudited live text'
    assert report['pages'][0]['source']==report['pages'][1]['source']=='native'
    assert len(failures)==1, 'Recovery must not rerun pixel recognition on an already blocked native unit'
    units=report['pages'][0]['degraded_units']
    assert len(units)==1 and units[0]['block'] is not None
    assert units[0]['reason']=='Synthetic native crop recognition failure'


@pytest.mark.parametrize('failure', [None, 'weak', 'different', 'distant', 'unavailable'])
def test_resampled_pixels_settle_only_the_same_supported_model_occurrence(failure):
    from cps.services.reflow import pixel_adjudication as pa
    source = words('The claar sample remains readable today.')
    model = 'The clear sample remains readable today.'.split()
    witness = pa.CropWitness(words('The clear sample remains readable today.', 39))
    views = [words('The clear sample remains readable today.', 96),
             words('The clear sample remains readable today.', 95)]
    if failure == 'weak':
        views[1][1]['confidence'] = 84
    elif failure == 'different':
        views[1][1]['text'] = 'claar'
    elif failure == 'distant':
        for word in views[1]:
            word['bbox'] = [v + 200 for v in word['bbox']]
    elif failure == 'unavailable':
        views = []
    witness.resampled = lambda: views
    result = adjudicate(model, source, witness)
    assert result['tokens'] == model
    assert result['marked'] == {1}, 'A model correction always retains its source mark.'
    decision = result['word_notes'][1]
    assert decision['status'] == ('RESOLVED' if failure is None else 'UNRESOLVED')
    if failure is None:
        assert decision['source'] == ['claar'] and decision['chosen'] == ['clear']
        assert decision['method'] == 'corroborated_resampled_pixel_readings'


def test_english_resampling_cannot_certify_a_changed_accent():
    from cps.services.reflow import pixel_adjudication as pa
    source = words('Please read cạlm beside the window.')
    witness = pa.CropWitness(words('Please read calm beside the window.', 39))
    witness.resampled = lambda: [words('Please read calm beside the window.'),
                                 words('Please read calm beside the window.')]
    result = adjudicate('Please read calm beside the window.'.split(), source, witness)
    assert result['word_notes'][2]['status'] == 'UNRESOLVED'


def test_two_corroborated_frames_can_settle_a_different_weak_primary_spelling():
    from cps.services.reflow import pixel_adjudication as pa
    source = words('The claar sample remains readable today.')
    model = 'The clear sample remains readable today.'.split()
    witness = pa.CropWitness(words('The lear sample remains readable today.', 39))
    witness.resampled = lambda: [words('The clear sample remains readable today.', 96),
                                 words('The clear sample remains readable today.', 95)]
    result = adjudicate(model, source, witness)
    assert result['tokens'] == model
    assert result['marked'] == {1}
    assert result['word_notes'][1]['status'] == 'RESOLVED'
    assert result['word_notes'][1]['method'] == 'corroborated_resampled_pixel_readings'


def test_resampling_cannot_override_a_confident_primary_spelling_conflict():
    from cps.services.reflow import pixel_adjudication as pa
    source = words('The claar sample remains readable today.')
    witness = pa.CropWitness(words('The lear sample remains readable today.', 96))
    witness.resampled = lambda: [words('The clear sample remains readable today.', 96),
                                 words('The clear sample remains readable today.', 95)]
    result = adjudicate('The clear sample remains readable today.'.split(), source, witness)
    assert result['word_notes'][1]['status'] == 'UNRESOLVED'


@pytest.mark.parametrize('transformed',[False,True])
def test_drawn_native_print_wins_over_conflicting_crop_ocr(tmp_path,monkeypatch,transformed):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa,vision_pixels
    class Package:
        def disclosure_image(self,*args):pass
    with pymupdf.open() as doc:
        leaf=doc.new_page(width=400,height=500)
        leaf.insert_text((30,70),'This does not change the clear sam-' if transformed else
                         'This does not change the clear sample.',fontsize=10)
        if transformed:leaf.insert_text((30,85),'ple.',fontsize=10)
        original=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in leaf.get_text('words')]
        witness=[dict(w,text='mot'if w['text']=='not'else w['text'])for w in original]
        calls=[]
        def recognize(*args):calls.append(True);return witness
        monkeypatch.setattr(pa,'read_crop',recognize)
        monkeypatch.setattr(vision_pixels,'words',lambda *args:witness)
        text='This does not change the clear sample.'
        page=dict(pno=0,body=['<p>'+text+'</p>'],asides=[])
        records=[dict(id='native-0',role='body',source_regions=[[25,55,390,90]])]
        vp._audit_native_pixels(doc,page,records,tmp_path,Package(),{})
    assert vp._seam_text(page['body'][0])==text
    assert page['asides']==[]
    assert not records[0].get('marked')
    assert records[0]['pixel_adjudication_status']=='VISIBLE_NATIVE_PRINT'
    assert calls==[], 'Visible mapped print is not a second OCR transcription'


@pytest.mark.parametrize('kind',['hidden','transparent','overpainted','unmapped'])
def test_unreliable_native_layer_does_not_earn_drawn_print_authority(kind):
    import pymupdf
    from cps.services.reflow import native_text
    with pymupdf.open()as doc:
        leaf=doc.new_page(width=300,height=200)
        leaf.insert_text((30,70),'A clear sample.',fontsize=10,
            render_mode=3 if kind=='hidden'else 0,fill_opacity=0 if kind=='transparent'else 1)
        if kind=='overpainted':
            from PIL import Image
            import io
            stream=io.BytesIO();Image.new('RGB',(300,200),'white').save(stream,format='PNG')
            leaf.insert_image(leaf.rect,stream=stream.getvalue())
        physical=[dict(text=w[4],bbox=list(w[:4]))for w in leaf.get_text('words')]
        if kind=='unmapped':physical[1]['text']='\ufffd'
        assert not native_text.drawn_word_authority(leaf,physical)


def test_one_alignment_disagreement_cannot_mark_every_confident_neighbor(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    printed='The clear sample remains readable beside the quiet window today.'
    physical=words(printed)
    # A weak secondary recognizer cannot establish uncertainty in agreed words.
    witness=words(printed,30)
    monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
    class Package:
        def image(self,*args):pass
        def disclosure_image(self,*args):pass
    with pymupdf.open()as doc:
        doc.new_page(width=400,height=300)
        _,record=vp.compile_page({'blocks':[block(printed,bbox=(0,0,1000,1000))]},
            [dict(w,text='x'+str(i))for i,w in enumerate(physical)],
            SourceDisplay(doc,0),0,Package(),{},pixel_words=physical,
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    assert record['blocks'][0]['marked']==[], 'No competing word or reader uncertainty exists'


def test_cropped_neighbor_failure_does_not_mark_an_exact_owned_word():
    # Context OCR loses the following short word, but this occurrence agrees.
    source=words('The clear sample remains readable today.')
    witness=[dict(w,confidence=96)for w in source if w['text']!='readable']
    result=adjudicate([w['text']for w in source],source,witness,uncertain=())
    assert not result['marked']
    uncertain=adjudicate([w['text']for w in source],source,witness,uncertain={3})
    assert 3 not in uncertain['marked'], 'Own exact crop proof is independent of its missing neighbor'
    assert 3 in uncertain['verified_unchanged']


@pytest.mark.parametrize(('printed','ocr'),[('each','cach'),('does','docs')])
def test_agreed_dictionary_word_is_not_rewritten_by_duplicate_ocr_settings(printed,ocr):
    from cps.services.reflow import pixel_adjudication as pa
    source=words('The reader '+printed+' compares a carefully printed sample.')
    pixels=[dict(w,text=ocr if w['text']==printed else w['text'])for w in source]
    result=pa.adjudicate([w['text']for w in source],source,pixels,corroboration=pixels)
    assert result['tokens']==[w['text']for w in source]
    # A credible competing reading may still receive its own unresolved control.
    notes=pa.audit_agreement(result['tokens'],source,pixels)
    assert notes[2]['alternative']==[ocr]
    assert notes[2]['status']=='UNRESOLVED'


def test_visible_native_words_remain_authoritative_in_a_vision_layout_unit(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    class Package:
        def image(self,*args):pass
        def disclosure_image(self,*args):pass
    with pymupdf.open()as doc:
        page=doc.new_page(width=400,height=300)
        page.insert_text((30,70),'The clear sample remains readable today.',fontsize=10)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        optical=[dict(w,text='readab1e'if w['text']=='readable'else w['text'])for w in source]
        monkeypatch.setattr(pa,'read_crop',lambda *args:optical)
        result,record=vp.compile_page({'blocks':[block('The clear sample readable today.',bbox=(0,0,1000,1000))]},source,
            SourceDisplay(doc,0),0,Package(),{},pixel_words=optical,
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    assert record['blocks'][0]['published_tokens']=='The clear sample remains readable today.'.split()
    assert all(d['status']=='RESOLVED'for d in record['blocks'][0]['adjudication'])
    assert record['blocks'][0]['marked']==[3], 'Only the real reader deletion earns a correction'


@pytest.mark.parametrize('rotation',[90,180,270])
def test_visible_native_print_authority_uses_unrotated_glyph_coordinates(tmp_path,monkeypatch,rotation):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    with pymupdf.open()as doc:
        leaf=doc.new_page(width=400,height=300)
        leaf.insert_text((30,70),'The clear sample remains readable today.',fontsize=10)
        leaf.set_rotation(rotation)
        display=SourceDisplay(doc,0)
        box=list(display.reading_rect([25,55,390,80]))
        monkeypatch.setattr(pa,'read_crop',lambda *args:pytest.fail('Native paint must remain authoritative'))
        page=dict(pno=0,body=['<p>The clear sample remains readable today.</p>'],asides=[])
        records=[dict(id='native-0',role='body',source_regions=[box])]
        vp._audit_native_pixels(doc,page,records,tmp_path,object(),{})
    assert records[0]['pixel_adjudication_status']=='VISIBLE_NATIVE_PRINT'
    assert page['asides']==[]


def test_unknown_legacy_face_cannot_make_a_latin_character_code_authoritative():
    import pymupdf
    from cps.services.reflow import native_text
    with pymupdf.open()as doc:
        page=doc.new_page(width=300,height=200)
        page.insert_text((30,70),'w',fontname='cour',fontsize=12)
        xref=page.get_fonts()[0][0]
        doc.xref_set_key(xref,'BaseFont','/CustomMarks')
        page=doc.reload_page(page)
        words=[dict(text=w[4],bbox=list(w[:4]))for w in page.get_text('words')]
        assert words[0]['text']=='w'
        assert not native_text.drawn_word_authority(page,words)


@pytest.mark.parametrize('recognition_available',[True,False])
@pytest.mark.parametrize('joined_wrap',[False,True])
def test_unproved_legacy_glyph_cannot_rewrite_reliable_neighbor_words(tmp_path,monkeypatch,recognition_available,joined_wrap):
    import pymupdf,re
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa,vision_pixels
    class Package:
        def disclosure_image(self,*args):pass
    with pymupdf.open()as doc:
        leaf=doc.new_page(width=300,height=200)
        leaf.insert_text((30,70),'not',fontname='helv',fontsize=12)
        leaf.insert_text((70,70),'w',fontname='cour',fontsize=12)
        if joined_wrap:leaf.insert_text((30,100),'sam-\nple',fontname='helv',fontsize=12)
        xref=next(f[0]for f in leaf.get_fonts()if f[3]=='Courier')
        doc.xref_set_key(xref,'BaseFont','/CustomMarks')
        leaf=doc.reload_page(leaf)
        original=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in leaf.get_text('words')]
        witness=[dict(w,text='mot'if w['text']=='not'else ')')for w in original]
        if not recognition_available:witness=[]
        monkeypatch.setattr(pa,'read_crop',lambda *args:witness)
        monkeypatch.setattr(vision_pixels,'words',lambda *args:witness)
        page=dict(pno=0,body=['<p>not w'+(' sample'if joined_wrap else'')+'</p>'],asides=[])
        records=[dict(id='native-0',role='body',source_regions=[[25,55,120,140]])]
        vp._audit_native_pixels(doc,page,records,tmp_path,Package(),{})
    assert vp._seam_text(re.sub(r'<a[^>]*class="source-disclosure-ref"[^>]*>.*?</a>','',page['body'][0]))==('not sample'if joined_wrap else'not')
    assert 'class="source-word-image"'in page['body'][0]
    assert records[0]['marked']==[1]
    assert len(page['asides'])==1


@pytest.mark.parametrize('recognition_available',[True,False])
def test_vision_layout_keeps_native_authority_local_to_each_font_occurrence(tmp_path,monkeypatch,recognition_available):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    class Package:
        def image(self,*args):pass
        def disclosure_image(self,*args):pass
    with pymupdf.open()as doc:
        leaf=doc.new_page(width=300,height=200)
        leaf.insert_text((30,70),'not',fontname='helv',fontsize=12)
        leaf.insert_text((70,70),'w',fontname='cour',fontsize=12)
        xref=next(f[0]for f in leaf.get_fonts()if f[3]=='Courier')
        doc.xref_set_key(xref,'BaseFont','/CustomMarks');leaf=doc.reload_page(leaf)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in leaf.get_text('words')]
        pixels=[dict(w,text='mot'if w['text']=='not'else ')')for w in source]
        if not recognition_available:pixels=[]
        monkeypatch.setattr(pa,'read_crop',lambda *args:pixels)
        _,record=vp.compile_page({'blocks':[block('not w',bbox=(0,0,1000,1000))]},source,
            SourceDisplay(doc,0),0,Package(),{},pixel_words=pixels,
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    assert record['blocks'][0]['published_tokens'][0]=='not'
    assert record['blocks'][0]['marked']==[1]
    assert record['blocks'][0]['source_image_words']==[1]


def test_abbreviated_font_trace_requires_every_matching_resource_to_be_authoritative():
    from cps.services.reflow import native_text
    face='StandardTextFace-Bold'
    assert native_text._face_authority({face+'Extended':True},face)
    assert not native_text._face_authority({face+'Extended':True,face+'Other':False},face)
    assert not native_text._face_authority({'ShortName':True},'Short')


@pytest.mark.parametrize('coverage',[1,.5])
def test_partial_image_overpaint_cannot_borrow_an_uncovered_neighbor_trace(coverage):
    import io,pymupdf
    from PIL import Image
    from cps.services.reflow import native_text
    with pymupdf.open()as doc:
        page=doc.new_page(width=300,height=200)
        page.insert_text((30,70),'clear sample',fontsize=12)
        words=[dict(text=w[4],bbox=list(w[:4]))for w in page.get_text('words')]
        picture=io.BytesIO();Image.new('RGB',(2,2),'white').save(picture,format='PNG')
        box=pymupdf.Rect(words[1]['bbox']);box.x1=box.x0+box.width*coverage
        page.insert_image(box,stream=picture.getvalue())
        assert native_text.drawn_word_authorities(page,words)==[True,False]


def test_unmapped_native_word_ink_keeps_its_separately_proved_note_action(tmp_path,monkeypatch):
    import pymupdf,re
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    class Package:
        def disclosure_image(self,*args):pass
    monkeypatch.setattr(pa,'read_crop',lambda *args:[])
    with pymupdf.open()as doc:
        leaf=doc.new_page(width=300,height=200)
        leaf.insert_text((30,70),'not',fontsize=12)
        leaf.insert_text((70,70),'w1',fontname='cour',fontsize=12)
        xref=next(f[0]for f in leaf.get_fonts()if f[3]=='Courier')
        doc.xref_set_key(xref,'BaseFont','/CustomMarks');leaf=doc.reload_page(leaf)
        label=next(c[3]for t in leaf.get_texttrace()for c in t['chars']if c[0]==ord('1'))
        page=dict(pno=0,body=['<p>not w<a id="origin" class="noteref" href="#note"><sup>1</sup></a></p>'],asides=[])
        images={}
        vp._audit_native_pixels(doc,page,[dict(id='native-0',role='body',source_regions=[[25,55,110,85]])],tmp_path,Package(),images)
    body=page['body'][0]
    assert body.count('id="origin"')==1
    assert body.count('href="#note"')==1
    assert body.count('class="source-word-image"')==1
    assert '<sup>1</sup>'in body
    assert next(iter(images.values()))['bbox'][2]<=label[0]+1e-3
    assert vp._seam_text(re.sub(r'<a[^>]*class="source-disclosure-ref"[^>]*>.*?</a>','',body))=='not 1'


def test_mapped_embedded_font_aliases_do_not_lose_native_character_authority():
    import pymupdf
    from cps.services.reflow import native_text
    assert native_text._face_authority({'Embedded Text Face Regular':True},'EmbeddedTextFace-Regular')
    assert not native_text._face_authority({'Embedded Text Face Regular':True,'EmbeddedTextFace-Regular':False},'Embedded Text Face Regular')
    with pymupdf.open()as doc:
        page=doc.new_page(width=300,height=200)
        page.insert_font(fontname='embedded',fontbuffer=pymupdf.Font('helv').buffer)
        page.insert_text((30,60),'clear sample',fontname='embedded')
        words=[dict(text=w[4],bbox=list(w[:4]))for w in page.get_text('words')]
        assert native_text.drawn_word_authority(page,words)


@pytest.mark.parametrize('legacy_initial',[False,True])
def test_native_decorative_initial_cannot_certify_a_truncated_word(tmp_path,monkeypatch,legacy_initial):
    import pymupdf,io
    from PIL import Image
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    class Package:
        def __init__(self):self.images={}
        def image(self,name,data):self.images[name]=data
        def disclosure_image(self,*args):pass
    package=Package();images={}
    monkeypatch.setattr(pa,'read_crop',lambda *args:[])
    with pymupdf.open()as doc:
        page=doc.new_page(width=400,height=300)
        face='cour'if legacy_initial else'helv'
        page.insert_text((30,110),'B',fontname=face,fontsize=36)
        if legacy_initial:
            xref=next(f[0]for f in page.get_fonts()if f[3]=='Courier')
            doc.xref_set_key(xref,'BaseFont','/CustomMarks');page=doc.reload_page(page)
        cap=page.get_text('words')[0]
        x=cap[2]+.4;y=cap[1]+12*1.075
        page.insert_text((x,y),'efore the quiet sample remains readable.',fontsize=12)
        page.insert_text((x,y+20),'The following row stays unchanged.',fontsize=12)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        body,record=vp.compile_page({'blocks':[block('Before the quiet sample remains readable. The following row stays unchanged.',bbox=(0,0,1000,1000))]},source,
            SourceDisplay(doc,0),0,package,images,pixel_words=[],
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
        own=record['blocks'][0]
        assert own['published_tokens']=='Before the quiet sample remains readable. The following row stays unchanged.'.split()
        assert own['marked']==([0]if legacy_initial else[])
        assert own['source_image_words']==([0]if legacy_initial else[])
        if legacy_initial:
            name=next(k for k in package.images if 'v4-d-'in k)
            picture=Image.open(io.BytesIO(package.images[name])).convert('L');clip=images[name]['bbox']
            # The compound original word must not reproduce the next row's ink.
            region=(int((x+3-clip[0])*picture.width/(clip[2]-clip[0])),
                    int((cap[1]+26-clip[1])*picture.height/(clip[3]-clip[1])),
                    int((x+20-clip[0])*picture.width/(clip[2]-clip[0])),
                    int((cap[1]+33-clip[1])*picture.height/(clip[3]-clip[1])))
            assert picture.crop(region).getextrema()[0]>235


def test_native_raised_ordinal_is_one_printed_word_without_a_correction(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    class Package:
        def image(self,*args):pass
        def disclosure_image(self,*args):pass
    monkeypatch.setattr(pa,'read_crop',lambda *args:[])
    with pymupdf.open()as doc:
        page=doc.new_page(width=300,height=200)
        page.insert_text((30,70),'1',fontsize=12)
        digit=page.get_text('words')[0]
        page.insert_text((digit[2]+.1,66),'st',fontsize=7)
        page.insert_text((55,70),'clear sample',fontsize=12)
        suffix=[c[3]for t in page.get_texttrace()if t['size']==7 for c in t['chars']]
        suffix_box=[min(b[0]for b in suffix),min(b[1]for b in suffix),max(b[2]for b in suffix),max(b[3]for b in suffix)]
        source=[dict(text='1',bbox=list(digit[:4]),confidence=99),dict(text='st',bbox=suffix_box,confidence=99)]+[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')if w[4]!='1st']
        _,record=vp.compile_page({'blocks':[block('1st clear sample',bbox=(0,0,1000,1000))]},source,
            SourceDisplay(doc,0),0,Package(),{},pixel_words=[],
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    assert record['blocks'][0]['published_tokens']==['1st','clear','sample']
    assert record['blocks'][0]['marked']==[]


@pytest.mark.parametrize('marker_size',[7,10])
def test_native_fused_raised_reference_cannot_become_a_second_body_number(tmp_path,monkeypatch,marker_size):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    class Package:
        def image(self,*args):pass
        def disclosure_image(self,*args):pass
    monkeypatch.setattr(pa,'read_crop',lambda *args:[])
    monkeypatch.setattr(vp,'_literal_note_refs',lambda *args,**kwargs:{})
    with pymupdf.open()as doc:
        page=doc.new_page(width=400,height=300)
        page.insert_text((30,70),'A clear sample',fontsize=12)
        physical=page.get_text('words');last=physical[-1]
        page.insert_text((last[2]+.1,65),'7',fontsize=marker_size)
        label=next(c[3]for t in page.get_texttrace()if t['size']==marker_size for c in t['chars'])
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in physical[:-1]]
        source.append(dict(text='sample7',bbox=[last[0],min(last[1],label[1]),label[2],max(last[3],label[3])],confidence=99))
        page.insert_text((30,160),'7 Words below remain readable.',fontsize=12)
        source.extend(dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')if w[1]>120)
        from cps.services.reflow.native_text import lexical_word_view
        assert lexical_word_view(page,source)==source, 'Raised digits stay literal without a requested note slot'
        def proof(tokens,*args,**kwargs):
            if '<n:n1>'not in tokens:return {}
            return {tokens.index('<n:n1>'):dict(status='UNPROVED',source_bbox=None,
                reason='synthetic_requested_slot_ambiguous',anchor_before_bbox=list(last[:4]),anchor_after_bbox=None)}
        monkeypatch.setattr(vp,'_placeholder_display_evidence',proof)
        context=dict(cache=tmp_path/'cache',labels={'7':[(0,'n1')]},owners={},
            catalogue={(0,'n1'):{'target':'v4-note-0-n1','label':'7'}},refs={})
        _,record=vp.compile_page({'blocks':[block('A clear sample<n:n1>',bbox=(0,0,1000,330)),block('Words below remain readable.',ident='n1',role='footnote',marker='7',bbox=(0,400,1000,1000))]},source,
            SourceDisplay(doc,0),0,Package(),{},pixel_words=[],note_context=context)
    assert record['blocks'][0]['published_tokens']==['A','clear','sample','<n:n1>']
    assert record['blocks'][0]['marked']==[]
    assert len(record['note_marker_disclosures'])==1


def test_native_raised_reference_serialization_cannot_discard_its_paragraph(tmp_path):
    import pymupdf
    from cps.services.reflow import assemble,build_epub,vision_pipeline as vp
    with pymupdf.open()as doc:
        leaf=doc.new_page(width=400,height=200)
        leaf.insert_text((30,70),'Ordinary sample,',fontsize=12)
        last=leaf.get_text('words')[-1]
        leaf.insert_text((last[2]+8,70),'remains readable.',fontsize=12)
        leaf.insert_text((last[2]+.1,65),'7',fontsize=7)
        # The renderer paints the reference after the body text in PDF object
        # order, while its physical origin remains inside the first line.
        runs=[('t','Ordinary sample,'),('sup',7),('t',' remains readable.')]
        element=assemble.Element('p',runs=runs,pno=0,bbox=[30,55,300,75])
        book=assemble.Book(elements=[element],pages={0:[element]})
        fragment=build_epub.page_fragment(book,0)
        native=dict(pno=0,body=[fragment],asides=[])
        records=vp._native_seam_blocks(doc,book,0,native)
        assert records[0]['native_occurrence_count']==1
        assert records[0]['source_regions']
        vp._audit_native_pixels(doc,native,records,tmp_path,object(),{})
    assert native['body']==[fragment]
    assert 'Ordinary sample,'in native['body'][0]
    assert 'remains readable.'in native['body'][0]
    assert 'noteref-unresolved'in native['body'][0]
    assert not records[0].get('marked')


def test_native_wrapped_symbol_expression_discloses_only_its_own_word(tmp_path,monkeypatch):
    import pymupdf,re
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    class Package:
        def __init__(self):self.images={}
        def image(self,name,data):self.images[name]=data
        def disclosure_image(self,*args):pass
    package=Package();images={}
    monkeypatch.setattr(pa,'read_crop',lambda *args:[])
    with pymupdf.open()as doc:
        leaf=doc.new_page(width=300,height=200)
        leaf.insert_text((30,70),'not',fontsize=12)
        leaf.insert_text((70,70),'x',fontname='cour',fontsize=12)
        leaf.insert_text((80,70),'-',fontsize=12)
        leaf.insert_text((30,100),'z.',fontname='cour',fontsize=12)
        leaf.insert_text((60,100),'sample',fontsize=12)
        xref=next(f[0]for f in leaf.get_fonts()if f[3]=='Courier')
        doc.xref_set_key(xref,'BaseFont','/CustomMarks');leaf=doc.reload_page(leaf)
        native=dict(pno=0,body=['<p>not x-z. sample</p>'],asides=[])
        records=[dict(id='native-0',role='body',source_regions=[[25,55,130,110]])]
        vp._audit_native_pixels(doc,native,records,tmp_path,package,images)
    body=native['body'][0]
    assert vp._seam_text(re.sub(r'<a[^>]*class="source-disclosure-ref"[^>]*>.*?</a>','',body))=='not sample'
    assert body.count('class="source-word-image"')==1
    assert records[0]['marked']==[1]
    assert records[0]['source_image_words']==[1]
    assert len(native['asides'])==1
    assert len(package.images)==1


@pytest.mark.parametrize('trace_decoding',['codes','replacement'])
@pytest.mark.parametrize('reading',['invented first forms','καλημέρα κόσμε forms'])
def test_unmapped_native_phrase_cannot_discard_its_proved_prose_neighbors(tmp_path,monkeypatch,reading,trace_decoding):
    import pymupdf,re
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    class Package:
        def disclosure_image(self,*args):pass
    monkeypatch.setattr(pa,'read_crop',lambda *args:[])
    if trace_decoding=='replacement':
        original=pymupdf.Page.get_texttrace
        def replacement_trace(page,*args,**kwargs):
            traces=original(page,*args,**kwargs)
            for trace in traces:
                if trace['font']=='CustomMarks':
                    trace['chars']=[(65533,*char[1:])if char[0]>32 else char for char in trace['chars']]
            return traces
        monkeypatch.setattr(pymupdf.Page,'get_texttrace',replacement_trace)
    with pymupdf.open()as doc:
        page=doc.new_page(width=400,height=250)
        page.insert_text((30,70),'The quiet',fontsize=12)
        page.insert_text((90,70),'x z',fontname='cour',fontsize=12)
        page.insert_text((130,70),'remains readable.',fontsize=12)
        xref=next(f[0]for f in page.get_fonts()if f[3]=='Courier')
        doc.xref_set_key(xref,'BaseFont','/CustomMarks');page=doc.reload_page(page)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        body,record=vp.compile_page({'blocks':[block('The quiet '+reading+' remains readable.',bbox=(0,0,1000,1000))]},
            source,SourceDisplay(doc,0),0,Package(),{},pixel_words=[],
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    own=record['blocks'][0]
    assert own['published_tokens']=='The quiet x z remains readable.'.split()
    assert own['source_image_words']==[2,3]
    assert own['marked']==[2,3]
    html=''.join(body['body'])
    assert html.count('class="source-word-image"')==2
    assert vp._seam_text(re.sub(r'<a[^>]*class="source-disclosure-ref"[^>]*>.*?</a>','',html))=='The quiet remains readable.'
    assert not any(word in html for word in reading.split())


@pytest.mark.parametrize('separate_columns',[False,True])
def test_partial_native_row_cannot_move_its_last_word_before_the_row(separate_columns):
    from cps.services.reflow import vision_pipeline as vp
    from tests.unit.test_reflow_vision_pilot import block
    import pymupdf
    class Display:
        rect=pymupdf.Rect(0,0,300,200)
    first=dict(text='Opening',bbox=[10,10,60,20])
    final=dict(text='ending',bbox=[92,29,130,41])
    lower=[dict(text='The',bbox=[10,32,30,42]),dict(text='clear',bbox=[35,32,65,42])]
    territories=[[0,0,140,32],[0,31,85,45]]
    if separate_columns:
        lower=[dict(w,bbox=[w['bbox'][0]+170,12,w['bbox'][2]+170,22])for w in lower]
        territories[1]=[170,0,260,45]
    b=block('Opening ending The clear'if separate_columns else'Opening The clear ending',bbox=(0,0,1000,1000))
    result=vp._owned_block_words(b,{'blocks':[b]},[first,final,*lower],Display(),{'b1':territories},ordered=True)
    assert [w['text']for w in result]==b['text'].split()


def test_native_print_at_a_partial_row_boundary_keeps_its_order_and_ink_extent(tmp_path,monkeypatch):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    class Package:
        def image(self,*args):pass
        def disclosure_image(self,*args):pass
    monkeypatch.setattr(pa,'read_crop',lambda *args:[])
    printed='The clear sample stays readable. Careful readers keep every word.'
    with pymupdf.open()as doc:
        page=doc.new_page(width=400,height=250)
        page.insert_text((30,60),'The clear sample stays readable.',fontsize=12)
        page.insert_text((30,90),'Careful readers keep every word.',fontsize=12)
        raw=page.get_text('words')
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in raw]
        last=source[-1]['bbox']
        territories=[[20,30,350,83],[20,84,last[0]-1,100]]
        body,record=vp.compile_page({'blocks':[block(printed,bbox=(0,0,1000,1000))]},source,
            SourceDisplay(doc,0),0,Package(),{},pixel_words=[],regions={'b1':territories},
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    own=record['blocks'][0]
    assert own['published_tokens']==printed.split()
    assert own['marked']==[]
    assert any(b[0]<=last[0] and b[1]<=last[1] and b[2]>=last[2] and b[3]>=last[3]
               for b in own['source_regions']), 'Stored native evidence must contain the actual last word ink'


@pytest.mark.parametrize('wrapped',[True,False])
def test_known_native_wrap_join_is_formatting_not_a_word_correction(tmp_path,monkeypatch,wrapped):
    import pymupdf
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import block
    class Package:
        def image(self,*args):pass
        def disclosure_image(self,*args):pass
    monkeypatch.setattr(pa,'read_crop',lambda *args:[])
    with pymupdf.open()as doc:
        page=doc.new_page(width=400,height=250)
        page.insert_text((30,60),'Please read care-',fontsize=12)
        x=30 if wrapped else page.get_text('words')[-1][2]+5
        y=90 if wrapped else 60
        page.insert_text((x,y),'fully.',fontsize=12)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in page.get_text('words')]
        _,record=vp.compile_page({'blocks':[block('Please read carefully.',bbox=(0,0,1000,1000))]},source,
            SourceDisplay(doc,0),0,Package(),{},pixel_words=[],
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    own=record['blocks'][0]
    assert own['published_tokens']==('Please read carefully.'.split()if wrapped else'Please read care- fully.'.split())
    if wrapped:assert own['marked']==[], 'Conserved letters joined across a printed line break are not a model rewrite'
    else:assert own['marked'], 'A real same-line space cannot be removed as a line-wrap normalization'


@pytest.mark.parametrize('interleaving_tail',[True,False])
@pytest.mark.parametrize('case',['complete','missing_fragment','wrong_box'])
def test_repeated_native_column_labels_bind_only_complete_factory_occurrences(tmp_path,case,interleaving_tail):
    import pymupdf
    from cps.services.reflow import assemble,build_epub,vision_pipeline as vp
    with pymupdf.open()as doc:
        page=doc.new_page(width=350,height=160)
        columns=[(30,'Clear,','sample'),(120,'Other','label'),(220,'Clear,','sample')]
        if interleaving_tail:columns.append((300,'Later','entry'))
        for x,first,last in columns:
            page.insert_text((x,70),first,fontsize=12)
            page.insert_text((x,85),last,fontsize=12)
        elements=[assemble.Element('p',runs=[('t','Clear, sample')],pno=0,bbox=[25,55,85,90]),
                  assemble.Element('p',runs=[('t','Clear, sample')],pno=0,
                    bbox=[115,55,180,90]if case=='wrong_box'else[215,55,280,90])]
        book=assemble.Book(elements=elements,pages={0:elements})
        native=build_epub._page_blocks({0:build_epub.page_fragment(book,0)})[0]
        if case=='missing_fragment':native['body']=native['body'][:1]
        original=list(native['body']);records=vp._native_seam_blocks(doc,book,0,native)
        # Row-major text interleaves the three columns. Repeated factory text
        # must bind its own proved element, not a global neighboring occurrence.
        if interleaving_tail:assert all(r['native_occurrence_count']==0 for r in records)
        if case=='complete':
            assert [r['source_regions']for r in records]==[[elements[0].bbox],[elements[1].bbox]]
            vp._audit_native_pixels(doc,native,records,tmp_path,object(),{})
            assert native['body']==original
            assert all(r['published_tokens']==['Clear,','sample']for r in records)
            assert all(not r.get('marked')and not r.get('degraded_source_context')for r in records)
        elif case=='missing_fragment':
            assert records[0]['source_regions']==[], 'Unequal factory/fragment counts cannot assign a repeated occurrence'
        else:
            assert records[0]['source_regions']==[elements[0].bbox]
            assert records[1]['source_regions']==[], 'An element box with different printed words cannot earn occurrence proof'

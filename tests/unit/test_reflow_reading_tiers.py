"""Synthetic reader disagreements retain assistance and adjacent original ink."""
import pytest
from xml.etree import ElementTree as ET
from cps.services.reflow import reading_tiers as tiers

pytestmark = pytest.mark.unit


def test_unproved_complete_reading_is_flagged_without_fabricating_a_resolution():
    original = dict(source=['modem'], model=['modern'], chosen=['modern'],
                    status='UNRESOLVED', method='unproved_shared_stroke_identity', source_image_only=True)
    result = dict(tokens=['modern'], marked={0}, word_notes={0: original})
    tiers.prepare(result)
    assert result['flagged_readings'][0]['reading'] == 'modern'
    assert not result['word_notes'][0]['source_image_only']
    assert original['source_image_only'] and original['status'] == 'UNRESOLVED'
    root = ET.fromstring(tiers.flagged_html('modern', 'images/word.png', 'own-note'))
    assert root.find('.//a').get('href') == '#own-note'
    assert root.find('.//img').get('src') == 'images/word.png'
    assert root.find('.//img').get('alt') == ''
    assert ''.join(root.itertext()).strip() == 'modern'


@pytest.mark.parametrize('method', ['unproved_native_character_map', 'incomplete_vertical_word_ink',
                                   'unproved_doubtful_leading_word_ink'])
def test_opaque_codes_or_missing_ink_do_not_become_guessed_assistance(method):
    choice = tiers.choose('garden', dict(status='UNRESOLVED', method=method, source=['garden'], model=['garden']))
    assert choice['tier'] == 'print' and choice['reading'] is None


def test_numeric_identifier_uses_complete_source_candidate_but_remains_flagged():
    choice = tiers.choose('7E42', dict(status='UNRESOLVED', source=['71E42'], model=['7E42']))
    assert choice == dict(tier='flagged', reading='71E42', method='source_complete_numeric_identifier')


def test_resolved_correction_keeps_its_existing_proof_and_notice():
    assert tiers.choose('modern', dict(status='RESOLVED'))['tier'] == 'proved'


@pytest.mark.parametrize('a,b,same', [('reader’s', "reader's", True), ('ﬁeld', 'field', True),
                                      ('co\u00adoperate', 'cooperate', True), ('cāre', 'care', False),
                                      ('modem', 'modern', False), ('O', '0', False)])
def test_only_representation_differences_are_typographically_equivalent(a, b, same):
    assert tiers.equivalent(a, b) is same


def test_real_page_compiler_keeps_plausible_reading_with_its_own_print(tmp_path, monkeypatch):
    import pymupdf
    from cps.services.reflow import pixel_adjudication as pa, vision_pipeline as vp
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import answer, block
    printed = 'The modern garden stays green.'
    with pymupdf.open() as drawn, pymupdf.open() as scan:
        pg = drawn.new_page(width=350, height=140)
        pg.insert_text((25, 60), printed, fontsize=12)
        words = [dict(text=w[4], bbox=list(w[:4]), confidence=95) for w in pg.get_text('words')]
        pg = scan.new_page(width=350, height=140)
        pg.insert_image(pg.rect, pixmap=drawn[0].get_pixmap())
        monkeypatch.setattr(pa, 'read_crop', lambda *a, **kw: pa.CropWitness(words))
        # This is a real unresolved occurrence, not a fabricated proof. The
        # rendering contract keeps its plausible reader and actual own image.
        original = pa.adjudicate
        def unsettled(*a, **kw):
            result = original(*a, **kw)
            result['marked'].add(1)
            result.setdefault('word_notes', {})[1] = dict(source=['modem'], model=['modern'],
                chosen=['modern'], status='UNRESOLVED', method='unproved_shared_stroke_identity',
                source_image_only=True, source_bbox=words[1]['bbox'])
            return result
        monkeypatch.setattr(pa, 'adjudicate', unsettled)
        class Package:
            def __init__(self): self.data = {}
            def image(self, name, data): self.data[name] = data
        package = Package(); images = {}
        page, record = vp.compile_page(answer(block(printed)), words, SourceDisplay(scan, 0), 0,
            package, images, pixel_words=words, regions={'b1': [[20, 40, 300, 70]]}, body_owners={},
            note_context={'cache': tmp_path/'cache', 'labels': {}, 'catalogue': {}, 'refs': {}})
    html = ''.join(page['body'])
    root = ET.fromstring('<root xmlns:epub="http://www.idpf.org/2007/ops">'+html+'</root>')
    flag = root.find('.//span[@class="reflow-flagged-reading"]')
    assert flag is not None and flag.find('span').text == 'modern'
    image = flag.find('.//img')
    assert image.get('src') in package.data
    assert image.get('alt') == ''
    assert 'vertical-align:-' in image.get('style')
    assert record['blocks'][0]['flagged_readings']['1']['tier'] == 'flagged'
    assert record['blocks'][0]['source_image_words'] == []


def test_numeric_identifier_cannot_be_a_silent_deleted_digit(tmp_path, monkeypatch):
    import pymupdf
    from cps.services.reflow import pixel_adjudication as pa, vision_pipeline as vp
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import answer, block
    printed='The printed position is 71E42 beside this marker.'
    with pymupdf.open() as drawn, pymupdf.open() as scan:
        p=drawn.new_page(width=400,height=150);p.insert_text((20,60),printed,fontsize=12)
        source=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in p.get_text('words')]
        pg=scan.new_page(width=400,height=150);pg.insert_image(pg.rect,pixmap=p.get_pixmap())
        wrong=[dict(w,text='7E42'if w['text']=='71E42'else w['text'])for w in source]
        witness=pa.CropWitness(wrong);witness.model_reading=True
        monkeypatch.setattr(pa,'read_crop',lambda *a,**kw:witness)
        class Package:
            def image(self,*a):pass
        page,record=vp.compile_page(answer(block(printed.replace('71E42','7E42'))),source,
            SourceDisplay(scan,0),0,Package(),{},pixel_words=source,regions={'b1':[[15,40,390,75]]},
            body_owners={},note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    b=record['blocks'][0]; index=4
    assert str(index) in b['flagged_readings']
    assert b['published_tokens'][index]=='71E42'
    assert b['flagged_readings'][str(index)]['method']=='source_complete_numeric_identifier'
    assert 'reflow-flagged-reading' in ''.join(page['body'])


def test_a_retained_first_read_without_counterevidence_is_assistance_not_a_proof():
    first=dict(status='RESOLVED',method='complete_vision_reading_without_optical_counterevidence',
               source=['modem'],model=['modern'],chosen=['modern'])
    second=dict(first,status='UNRESOLVED',method='unproved_shared_stroke_identity')
    assert tiers.choose('modern',first)['tier']=='flagged'
    assert tiers.choose('modern',second)['tier']=='flagged'


@pytest.mark.parametrize('source,model,settled',[("reader’s","reader's",True),('ﬁeld','field',True),
    ('cāre','care',False),('modem','modern',False),('O','0',False)])
def test_representation_equivalence_removes_doubt_without_erasing_correction_notice(source,model,settled):
    result=dict(tokens=[model],marked={0},word_notes={0:dict(source=[source],model=[model],chosen=[model],
        status='UNRESOLVED',method='unproved_quote_identity',source_image_only=True)},guard=[])
    tiers.settle_typographic_equivalents(result)
    tiers.prepare(result)
    assert (result['word_notes'][0]['status']=='RESOLVED') is settled
    assert result['marked']=={0}
    if settled:assert not result['flagged_readings'] and not result['word_notes'][0]['source_image_only']


@pytest.mark.parametrize('source,model,expected',[('garden','gardem','garden'),
    ('modem','modern','modern'),('reader','Ravena','Ravena'),('cāre','care','care')])
def test_assistance_ranks_observed_lexical_candidates_without_certifying_them(source,model,expected):
    decision=dict(source=[source],model=[model],chosen=[model],status='UNRESOLVED',method='unsettled_alternate_pixel_readings')
    choice=tiers.choose(model,decision)
    assert choice['tier']=='flagged' and choice['reading']==expected
    assert decision['status']=='UNRESOLVED' and decision['chosen']==[model]


def assert_flagged_print(markup, readings=None):
    """Every plausible run keeps live words and linked inline or shared print."""
    root = ET.fromstring('<root xmlns:epub="http://www.idpf.org/2007/ops">'+markup+'</root>')
    flags = [n for n in root.iter() if n.get('class') == 'reflow-flagged-reading']
    assert flags
    actual = []
    for flag in flags:
        live = next(n for n in flag if n.get('class') == 'reflow-plausible')
        reading = ''.join(live.itertext())
        assert reading.strip()
        links = [n for n in flag.iter() if n.get('class') == 'source-word-image']
        if links:
            assert len(links) == 1
            link = links[0]
        else:
            # Dense or refused inline evidence has one explicit unit print.
            units = [n for n in root.iter() if n.get('class') == 'reflow-unit-print']
            assert len(units) == 1
            link = units[0][0]
            assert link.get('class') == 'source-disclosure-ref'
            assert len(link) == 2 and link[1].tag == 'sup' and link[1].text == '†'
        assert link.get('href', '').startswith('#v4-')
        assert link[0].tag == 'img'
        assert link[0].get('src', '').startswith('images/') and link[0].get('alt') == ''
        actual.append(reading)
    if readings is not None:
        assert ' '.join(actual).split() == readings
    return root, flags


def test_native_assistance_keeps_its_first_reading_without_lexical_overrule():
    decision=dict(source=['remalns'],model=['remalns'],alternative=['remains'],
                  status='UNRESOLVED',method='pixel_read_inconclusive')
    choice=tiers.choose('remalns',decision,rank_lexical=False)
    assert choice['tier']=='flagged' and choice['reading']=='remalns'


@pytest.mark.parametrize('prove_one',[False,True])
def test_compiler_shares_one_legible_print_for_a_dense_multiline_reading(tmp_path,monkeypatch,prove_one):
    import pymupdf
    from cps.services.reflow import pixel_adjudication as pa,vision_pipeline as vp
    from cps.services.reflow.source_display import SourceDisplay
    from tests.unit.test_reflow_vision_pilot import answer,block
    lines=['The garden grows green.','The clear sample remains readable.']
    printed=' '.join(lines)
    with pymupdf.open()as drawn,pymupdf.open()as scan:
        pg=drawn.new_page(width=350,height=180)
        for y,line in zip((60,90),lines):pg.insert_text((30,y),line,fontsize=16)
        words=[dict(text=w[4],bbox=list(w[:4]),confidence=95)for w in pg.get_text('words')]
        pg=scan.new_page(width=350,height=180);pg.insert_image(pg.rect,pixmap=drawn[0].get_pixmap())
        monkeypatch.setattr(pa,'read_crop',lambda *a,**kw:pa.CropWitness(words))
        original=pa.adjudicate
        def unsettled(*a,**kw):
            result=original(*a,**kw);result['adjudication']=[]
            result['marked']=set(range(len(result['tokens'])))
            result['word_notes']={i:dict(source=[word],model=[word],chosen=[word],
                status='UNRESOLVED',method='pixel_read_inconclusive',source_bbox=words[i]['bbox'])
                for i,word in enumerate(result['tokens'])}
            if prove_one:result['word_notes'][2].update(status='RESOLVED',method='independent_upright_crop')
            return result
        monkeypatch.setattr(pa,'adjudicate',unsettled)
        class Package:
            def __init__(self):self.data={}
            def image(self,name,data):self.data[name]=data
        package=Package();images={}
        page,record=vp.compile_page(answer(block(printed)),words,SourceDisplay(scan,0),0,
            package,images,pixel_words=words,regions={'b1':[[25,40,310,100]]},body_owners={},
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    expected=[word for i,word in enumerate(printed.split())if not(prove_one and i==2)]
    root=ET.fromstring('<root xmlns:epub="http://www.idpf.org/2007/ops">'+''.join(page['body'])+'</root>')
    flags=[n for n in root.iter()if n.get('class')=='reflow-flagged-reading']
    assert ' '.join(''.join(n.find('span').itertext())for n in flags).split()==expected
    def visible(node):
        if 'source-disclosure-ref' in node.get('class','').split():return ''
        return (node.text or '')+''.join(visible(child)+(child.tail or '') for child in node)
    assert visible(root).split()==printed.split(),'A proved neighbor cannot hide a plausible word'
    assert all(flag.find('.//img')is None for flag in flags)
    unit=root.find('.//div[@class="reflow-unit-print"]')
    assert unit is not None and len(list(root.iter('sup')))==1+int(prove_one)
    image=unit.find('.//img');metadata=images[image.get('src')]
    assert metadata['kind']=='flagged_unit_preview'
    assert metadata['reason']=='dense_flagged_reading'
    assert metadata['bbox'][1]<=words[0]['bbox'][1]
    assert metadata['bbox'][3]>=words[-1]['bbox'][3]
    assert image.get('style')is None
    assert '.reflow-unit-print img { max-width:95%; width:auto; height:auto;' in tiers.stylesheet(), 'Several printed lines retain their readable aspect ratio'


def test_appended_print_has_named_evidence_link_without_repeating_the_reading():
    root = ET.fromstring(tiers.flagged_html('modern', 'images/word.png', 'own-note'))
    link = root.find('a')
    assert link.get('title') == 'Inspect original print'
    assert link.get('href') == '#own-note'
    assert link.find('img').get('alt') == ''
    assert ''.join(root.itertext()) == 'modern'


@pytest.mark.parametrize('first,source', [('cāre', 'care'), ('garden’s', 'gardens')])
def test_dictionary_ranker_cannot_erase_a_reader_diacritic_or_possessive(first, source, monkeypatch):
    from cps.services.reflow import text_quality
    monkeypatch.setattr(text_quality, 'english_lexicon', lambda: {'care', 'gardens'})
    choice = tiers.choose(first, dict(status='UNRESOLVED', method='unproved_quote_identity',
                                     source=[source], model=[first]))
    assert choice['tier'] == 'flagged'
    assert choice['reading'] == first

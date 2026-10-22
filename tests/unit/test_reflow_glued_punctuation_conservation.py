"""Synthetic punctuation belongs to one occurrence across whitespace changes."""
import pytest
from cps.services.reflow.reading_tiers import conserve_punctuation_boundaries

pytestmark=pytest.mark.unit


@pytest.mark.parametrize('glyph',[',','.',';',':'])
@pytest.mark.parametrize('case',['glued','two_printed','decimal','remote'])
def test_correction_cannot_repeat_the_delimiter_already_moved_to_previous_token(glyph,case):
    source=[dict(text='garden'+(glyph if case=='two_printed'else''),bbox=[10,20,40,32]),
            dict(text='.'+('25'if case=='decimal'else'which'),bbox=[44,20,70,32])]
    if case=='remote':source[1]['bbox']=[44,80,70,92]
    model=['garden'+glyph,'25'if case=='decimal'else'which']
    token=glyph+model[1]
    result=dict(tokens=[model[0],token],marked={1},word_notes={1:dict(status='RESOLVED',
        source_range=[1,2],model_range=[1,2])},guard=[])
    conserve_punctuation_boundaries(result,source,model)
    assert result['tokens']==[model[0],model[1]if case=='glued'else token]
    assert (1 in result['marked']) is (case!='glued')
    assert bool(result['guard']) is (case=='glued')

"""Synthetic publication-text controls with independently authored word lists."""
import pytest

from cps.services.reflow import text_quality as q

pytestmark = pytest.mark.unit
CLEAR = 'Green baskets contain ripe apples while purple carts carry fresh bread'
LEXICON = frozenset(CLEAR.lower().split()) | {'container', 'paintwork', 'threading'}


def convergent_damage():
    # Forty words; three recognized spellings have repeated internal case flips.
    known = ('coNtAiner paInTwork thReAding ' + ' '.join(CLEAR.split() * 3)).split()[:30]
    unknown = ['zqx' + chr(97 + n) for n in range(10)]
    return ' '.join(known + unknown)


def test_convergent_subthreshold_signals_reject_a_damaged_reading():
    metric = q.inspect(convergent_damage(), lexicon=LEXICON)
    assert metric['latin_words'] == 40
    assert metric['dictionary_miss_rate'] == .25
    assert metric['case_flip_rate'] == .075
    assert metric['worst_window_miss_rate'] == .625
    assert metric['severe_case_words'] == 0
    assert metric['needs_read']
    assert 'combined_case_dictionary_window' in metric['reasons']
    assert q.inspect_disclosure(convergent_damage(), lexicon=LEXICON)['needs_read']


def test_recognized_case_variants_and_unfamiliar_normal_case_are_independent_controls():
    assert not q.inspect(' '.join(CLEAR.split() * 5), lexicon=LEXICON)['needs_read']
    assert not q.inspect(convergent_damage().lower(), lexicon=LEXICON)['needs_read']
    assert not q.inspect(convergent_damage(), lexicon=LEXICON | {'zqx' + chr(97+n) for n in range(10)})['needs_read']
    assert not q.inspect('BrightForms maker joins CedarWorks today', lexicon={'maker','joins','today'})['needs_read']


def test_partitioned_failure_preserves_an_unrelated_clean_unit():
    metric = q.inspect_partitioned([('clear', CLEAR), ('damaged', convergent_damage())], lexicon=LEXICON)
    assert metric['needs_read']
    assert metric['rejected_units'] == ['damaged']
    assert not metric['units']['clear']['needs_read']
    assert metric['units']['damaged']['needs_read']


def test_repeated_owner_does_not_reorder_a_cross_unit_window():
    unknown = ['zqx' + chr(97+n) for n in range(12)]
    units = [('left', ' '.join(unknown[:6])), ('middle', ' '.join(unknown[6:])), ('left', CLEAR)]
    metric = q.inspect_partitioned(units, lexicon=LEXICON)
    assert metric['needs_read']
    assert metric['rejected_units'] == ['left', 'middle']


def test_window_crossing_units_is_attributed_to_its_misses_only():
    words = ['zqx' + chr(97+n) for n in range(12)]
    units = [('clear', CLEAR), ('left', ' '.join(words[:6])), ('right', ' '.join(words[6:]))]
    metric = q.inspect_partitioned(units, lexicon=LEXICON)
    assert metric['needs_read']
    assert metric['rejected_units'] == ['left', 'right']
    assert not metric['units']['clear']['needs_read']


@pytest.mark.parametrize('markup', [
    '<img alt="{text}" src="figure.png"/>',
    '<aside class="vision-disclosure">{text}</aside>',
    '<p class="source-reading-annotation">{text}</p>',
    '<a aria-label="{text}" href="#detail">{clear}</a>',
    '<p title="{text}">{clear}</p>',
])
def test_every_reader_received_text_channel_is_swept(markup):
    page = dict(pno=0, body=['<p>'+CLEAR+'</p>', markup.format(text=convergent_damage(), clear=CLEAR)], asides=[])
    before = q.reading_text(page)
    published = q.publication_text(page)
    assert convergent_damage() in published
    sweep = q.sweep_publication_pages([page], {0:dict(source='vision')}, [0], lexicon=LEXICON)
    assert not sweep['passed'] and sweep['pages'][0]['needs_read']
    with pytest.raises(ValueError, match='Every source page'):
        q.validate_sweep(sweep, [0])
    assert q.reading_text(page) == before


def test_short_image_description_does_not_hide_behind_the_body():
    page = dict(pno=0, body=['<p>'+(CLEAR+' ')*20+'</p>', '<img alt="zqxone zqxtwo zqxthree"/>'], asides=[])
    sweep = q.sweep_publication_pages([page], {0:dict(source='vision')}, [0], lexicon=LEXICON)
    assert not sweep['passed']
    assert any('short_disclosure_dictionary_misses' in m['reasons'] for m in sweep['pages'][0]['units'].values())


def test_publication_projection_keeps_inline_words_and_excludes_nonreading_code():
    page = dict(pno=0, body=['<p>con<span>tain</span>er</p><script>zqxone zqxtwo zqxthree</script>',
        '<style>zqxone zqxtwo zqxthree</style>', '<p>'+CLEAR+'</p>'], asides=[])
    parts = q.publication_parts(page)
    assert any(part['text'] == 'container' for part in parts)
    assert 'zqxone' not in q.publication_text(page)
    sweep = q.sweep_publication_pages([page], {0:dict(source='native')}, [0], lexicon=LEXICON)
    assert sweep['passed']
    q.validate_sweep(sweep, [0])

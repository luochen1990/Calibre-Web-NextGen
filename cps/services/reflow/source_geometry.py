"""Corroborate source ownership using coherent, unique original word rows.

Accent folding is only a placement comparison. This module never changes the
chosen text or manufactures source words, glyphs, or confidence.
"""
from difflib import SequenceMatcher
import statistics
import unicodedata


def _key(value):
    value = unicodedata.normalize('NFKD', value.casefold())
    return ''.join(char for char in value
                   if not unicodedata.combining(char) and (char.isalnum() or char == '_'))


def unique_anchors(optical, available, candidate):
    """Return a uniquely corroborated compact row window, or no proof.

    A distant common word cannot bridge unrelated entries. Adjacent printed
    rows may form one passage; a blank vertical interval or column gap starts
    another. Competing equally strong physical occurrences remain ambiguous.
    Returned indices identify original word instances, never interpolated boxes.
    """
    available = list(available)
    keys = [_key(token) for token in candidate]
    positions = {j for j, token in enumerate(keys) if token}
    if not available or not positions:
        return []
    source_keys = {i: _key(optical[i]['text']) for i in available}
    wanted = set(keys) - {''}
    height = max(1, statistics.median(
        optical[i]['bbox'][3] - optical[i]['bbox'][1] for i in available))
    rows = []
    center = lambda i: (optical[i]['bbox'][1] + optical[i]['bbox'][3]) / 2
    for index in sorted(available, key=lambda i: (center(i), optical[i]['bbox'][0])):
        row = next((row for row in reversed(rows)
                    if abs(row[0] - center(index)) <= height * .55), None)
        if row is None:
            rows.append([center(index), [index]])
        else:
            row[1].append(index)
            row[0] = statistics.median(center(i) for i in row[1])
    pieces = []
    for _, row in rows:
        part = []
        for index in sorted(row, key=lambda i: optical[i]['bbox'][0]):
            if part and optical[index]['bbox'][0] - optical[part[-1]]['bbox'][2] > height * 4:
                pieces.append(part)
                part = []
            part.append(index)
        if part:
            pieces.append(part)
    pieces.sort(key=lambda row: (min(optical[i]['bbox'][1] for i in row),
                                 optical[row[0]]['bbox'][0]))
    proofs = {}
    for start, first in enumerate(pieces):
        if not any(source_keys[i] in wanted for i in first):
            continue
        sequence = []
        last = first
        for part in pieces[start:]:
            left = min(optical[i]['bbox'][0] for i in part)
            right = max(optical[i]['bbox'][2] for i in part)
            if sequence:
                old_left = min(optical[i]['bbox'][0] for i in last)
                old_right = max(optical[i]['bbox'][2] for i in last)
                gap = min(optical[i]['bbox'][1] for i in part) - max(optical[i]['bbox'][3] for i in last)
                if gap < height * .25:
                    # A second column on the same row is a separate window.
                    continue
                if gap > height * 2.5 or left > old_right + height or right < old_left - height:
                    break
            sequence += part
            last = part
            if len(sequence) > max(20, len(keys) * 2):
                break
            if not any(source_keys[i] in wanted for i in part):
                continue
            matcher = SequenceMatcher(None, [source_keys[i] for i in sequence], keys,
                                      autojunk=False)
            anchors = [(sequence[i], j) for a, b, count in matcher.get_matching_blocks()
                       for i, j in zip(range(a, a + count), range(b, b + count)) if keys[j]]
            covered = {j for _, j in anchors}
            if len(covered) < min(5, len(positions)) or len(covered) / len(positions) < .6:
                continue
            # Complete short readings are required: common suffixes alone do
            # not locate an unfamiliar name or abbreviated heading.
            if len(positions) <= 4 and covered != positions:
                continue
            identity = tuple(anchors)
            longest = max(count for _, _, count in matcher.get_matching_blocks())
            # Extra context in a larger window cannot defeat an equally exact
            # second occurrence. The physical anchor identity is the proof.
            score = (len(covered), longest)
            if identity not in proofs or score > proofs[identity]:
                proofs[identity] = score
    if not proofs:
        return []
    best = max(proofs.values())
    winners = [identity for identity, score in proofs.items() if score == best]
    return list(winners[0]) if len(winners) == 1 else []


def labelled_note_anchors(optical, available, candidate, marker):
    """Locate repeated note wording only through its own printed opening label.

    A hint's proximity cannot choose between repeated citations. A literal
    source label must stand left of the first printed row, and the chosen
    passage must start on that row. The next aligned printed label bounds the
    available continuation. Unreadable or competing labels prove no placement.
    """
    import re
    available = list(available)
    marker = marker.strip().strip('().[] ')
    if not re.fullmatch(r'\d{1,4}|[*†‡]', marker) or not available:
        return None
    label_text = lambda word: word['text'].strip().strip('().[] ')
    sizes = [optical[i]['bbox'][3] - optical[i]['bbox'][1] for i in available]
    height = max(1, statistics.median(sizes))
    center = lambda box: (box[1] + box[3]) / 2
    proofs = []
    for index in available:
        label = optical[index]
        if label_text(label) != marker:
            continue
        box = label['bbox']
        # Printed labels occupy the opening margin. The very next physical
        # word must be adjacent and share the original baseline.
        row = [i for i in available if i != index and
               abs(center(optical[i]['bbox']) - center(box)) <= height * .75 and
               optical[i]['bbox'][0] >= box[2]]
        row.sort(key=lambda i: optical[i]['bbox'][0])
        if not row or row[0] == index:
            continue
        if not 0 <= optical[row[0]]['bbox'][0] - box[2] <= height * 4:
            continue
        right = optical[row[0]]['bbox'][2]
        for previous, current in zip(row, row[1:]):
            if optical[current]['bbox'][0] - optical[previous]['bbox'][2] > height * 4:
                break
            right = optical[current]['bbox'][2]
        endings = [center(optical[i]['bbox']) for i in available if i != index and
                   re.fullmatch(r'\d{1,4}|[*†‡]', label_text(optical[i])) and
                   abs(optical[i]['bbox'][0] - box[0]) <= height and
                   center(optical[i]['bbox']) > center(box) + height * .8]
        bottom = min(endings, default=float('inf')) - height * .5
        pool = [i for i in available if i != index and
                center(box) - height * .75 <= center(optical[i]['bbox']) < bottom and
                optical[i]['bbox'][0] >= box[2] and
                optical[i]['bbox'][2] <= right + height * 4]
        anchors = unique_anchors(optical, pool, candidate)
        if not anchors:
            continue
        first = min(anchors, key=lambda pair: pair[1])
        if first[1] > 2 or abs(center(optical[first[0]]['bbox']) - center(box)) > height * .75:
            continue
        proofs.append(dict(anchors=anchors, marker_bbox=list(box),
                           first_word_bbox=list(optical[row[0]]['bbox'])))
    return proofs[0] if len(proofs) == 1 else None

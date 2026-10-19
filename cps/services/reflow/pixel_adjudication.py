"""Settle individual transcription edits using an independent pixel witness.

A native layer and a model can disagree without either one being authoritative.
Only a uniquely context-bound reading from the second recognition can resolve an
edit. Unresolved edits retain their source-linked uncertainty; dictionary words
alone never certify a substitution, omission or insertion.
"""
from difflib import SequenceMatcher
import unicodedata


def _key(value):
    return unicodedata.normalize('NFKC', value).replace('\u00ad', '').translate(
        str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'"}))


def _accent_conflict(left,right):
    """An English pixel read cannot prove a different diacritic on equal letters."""
    def letters(value):
        return ''.join(c for c in unicodedata.normalize('NFD',value.casefold())
                       if c.isalnum() or unicodedata.combining(c))
    a,b=letters(left),letters(right)
    bare=lambda value:''.join(c for c in value if not unicodedata.combining(c))
    return a!=b and bare(a)==bare(b)


def _ordinal(value):
    import re
    match=re.fullmatch(r'([0-9]+)(st|nd|rd|th)[.,;:!?]?',_key(value))
    if not match:return False
    number=int(match[1]);suffix=('th'if 10<=number%100<=20 else {1:'st',2:'nd',3:'rd'}.get(number%10,'th'))
    return match[2]==suffix


def _ordinal_noise(value):
    import re
    return bool(re.fullmatch(r'[0-9gGsSiIlLoOuUbBrRaA]+(?:st|nd|rd|th)[.,;:!?]?',_key(value)))


def _own_word_choice(source,witness):
    reader=getattr(witness,'word_readings',None)
    if reader is None:return None
    views=reader(source)
    if len(views)!=2:return None
    own=[[w for w in view if _complete_word_occurrence(source,w)]for view in views]
    if not all(len(row)==1 and row[0].get('confidence',0)>=85 for row in own):return None
    first,second=own[0][0],own[1][0]
    return first if _key(first['text'])==_key(second['text'])else None


def _boundary_parts(model,source):
    """Insert proved spaces while preserving every original model codepoint."""
    parts=[];start=cursor=0
    for word in source[:-1]:
        count=sum(c.isalnum() for c in unicodedata.normalize('NFKD',word))
        seen=0
        while cursor<len(model) and seen<count:
            seen+=sum(c.isalnum() for c in unicodedata.normalize('NFKD',model[cursor]))
            cursor+=1
        while cursor<len(model) and unicodedata.combining(model[cursor]):cursor+=1
        parts.append(model[start:cursor]);start=cursor
    parts.append(model[start:])
    return parts


def _matches(words, sequence):
    if not sequence:return []
    return [i for i in range(len(words)-len(sequence)+1)
            if words[i:i+len(sequence)] == sequence]


def _same_occurrence(left,right):
    a=left['bbox'];b=right['bbox']
    height=max(a[3]-a[1],b[3]-b[1],1)
    return (abs((a[1]+a[3]-b[1]-b[3])/2)<=height*.75+2
            and min(a[3],b[3])>max(a[1],b[1])
            and min(a[2],b[2])-max(a[0],b[0])>=.3*min(a[2]-a[0],b[2]-b[0]))


def blank_between(display,left,right,regions):
    """Prove absence only from complete original intervening ink, not OCR gaps."""
    from . import extract
    a=left['bbox'];b=right['bbox']
    height=min(a[3]-a[1],b[3]-b[1])
    if min(a[3],b[3])-max(a[1],b[1])<height*.5 or b[0]<=a[2]:return False
    # Concatenated disjoint crops can exclude the very word under dispute.
    envelope=[a[0],min(a[1],b[1]),b[2],max(a[3],b[3])]
    if not any(r[0]<=envelope[0] and r[1]<=envelope[1] and r[2]>=envelope[2] and r[3]>=envelope[3] for r in regions):return False
    box=extract.pymupdf.Rect(a[2]+.1,envelope[1],b[0]-.1,envelope[3])
    if box.width<=0:return False
    pix=display.pixmap(box,scale=4,max_pixels=200_000)
    from PIL import Image
    import io
    gray=Image.open(io.BytesIO(pix.tobytes('png'))).convert('L')
    return gray.getextrema()[0]>=180


def _word_note(record, index, span_id):
    """Keep full edit evidence once; never invent a positional word pairing."""
    if max(len(record[key]) for key in ('source','model','chosen'))<=1:return record
    if (record['chosen']==record['source'] and not record['model'] or
            not record['source'] and record['chosen']==record['model']):
        return dict(record,source=record['source'][index:index+1],
                    model=record['model'][index:index+1],chosen=record['chosen'][index:index+1])
    return dict(source=[],model=[],chosen=record['chosen'][index:index+1],
        status=record['status'],method=record['method'],mapping='unplaced_within_span',
        span_id=span_id,source_range=record.get('source_range'),model_range=record.get('model_range'))


def _terminal_word_view(model, words):
    """Keep split terminal punctuation in its existing lexical token.

    This view changes neither the literal punctuation nor its original input
    records. A distant punctuation instance cannot enlarge the word's source
    box or establish physical continuation proof. The original word inventory
    remains authoritative for that separate geometry decision.
    """
    literals={_key(token) for token in model}
    result=[]
    for word in words:
        suffix=word['text']
        if (result and suffix and all(c in '.,;:!?' for c in suffix)
                and _key(result[-1]['text']+suffix) in literals):
            previous=result[-1]
            a,b=previous['bbox'],word['bbox']
            height=max(a[3]-a[1],b[3]-b[1],1)
            adjacent=(min(a[3],b[3])-max(a[1],b[1])>=height*.5
                      and 0<=b[0]-a[2]<=height*1.5)
            box=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])] if adjacent else a
            result[-1]=dict(previous,text=previous['text']+suffix,bbox=box,
                confidence=min(previous.get('confidence',100),word.get('confidence',100)))
        else:result.append(word)
    if any(hasattr(words,name) for name in ('alternates','resampled','word_readings')):
        result=CropWitness(result)
        for name in ('alternates','resampled','mark_absence','word_readings'):
            if hasattr(words,name):setattr(result,name,getattr(words,name))
    return result


def proved_native_occurrence(token,box,words):
    if box is None:return False
    return len([w for w in words if _key(token)==_key(w['text']) and
                _same_occurrence(dict(bbox=box),w)])==1


def adjudicate(model, source, witness, uncertain=(), absence_proof=None, corroboration=(), unresolved_source=False, source_print=False, native_print_words=()):
    """Return word-level decisions; ``witness`` is independent crop OCR.

    Empty alternatives require both adjacent anchors, so a missing edge cannot
    prove a deletion. Repeated phrases require a unique surrounding sequence.
    Source rectangles belong to occurrences, rather than spelling lookups.
    """
    source=_terminal_word_view(model,source)
    witness=_terminal_word_view(model,witness)
    corroboration=_terminal_word_view(model,corroboration)
    source_tokens=[w['text'] for w in source]
    visible=[(i,t) for i,t in enumerate(model) if not t.startswith('<n:')]
    proposed=[t for _,t in visible]
    original=[_key(t) for t in source_tokens]
    candidate=[_key(t) for t in proposed]
    pixels=[_key(w['text']) for w in witness]
    changes=[];chosen=[];decisions=[];pending=set(uncertain)
    matcher=SequenceMatcher(None, original, candidate, autojunk=False)
    placement=lambda t:''.join(c for c in unicodedata.normalize('NFKD',t.casefold())
                               if c.isalnum())
    operations=[]
    for kind,a,z,b,y in matcher.get_opcodes():
        if kind!='replace' or max(z-a,y-b)>32:
            operations.append((kind,a,z,b,y));continue
        folded=SequenceMatcher(None,[placement(t) for t in original[a:z]],
            [placement(t) for t in candidate[b:y]],autojunk=False)
        for tag,i,k,j,l in folded.get_opcodes():
            # Similarity proposes correspondence, never the printed word.
            # The independent physical occurrence must still prove a choice.
            paired=(k-i==l-j and (tag=='equal' or all(
                SequenceMatcher(None,placement(original[a+n]),placement(candidate[b+m])).ratio()>=.65
                for n,m in zip(range(i,k),range(j,l)))))
            if paired:
                operations.extend(('replace',a+n,a+n+1,b+m,b+m+1)
                                  for n,m in zip(range(i,k),range(j,l)))
            else:operations.append(('replace',a+i,a+k,b+j,b+l))
    for kind,a,z,b,y in operations:
        if source_print or a<z and all(proved_native_occurrence(w['text'],w['bbox'],native_print_words)for w in source[a:z]):
            # This occurrence is reliably decoded visible native paint, even
            # when a page uses the vision route for layout or other units.
            # Recognition settings cannot turn punctuation into words, remove
            # authored text, or change its spelling. The first reader's real
            # lexical edits remain recorded one word at a time.
            lexical=any(c.isalnum()for token in source_tokens[a:z]+proposed[b:y] for c in token)
            changed=kind!='equal' and lexical
            record=dict(source=source_tokens[a:z],model=proposed[b:y],chosen=source_tokens[a:z],
                status='RESOLVED',method='visible_mapped_native_glyphs',source_range=[a,z],model_range=[b,y])
            if changed:decisions.append(record)
            for j,word in enumerate(source[a:z]):
                note=(dict(record,source=[word['text']],chosen=[word['text']],
                    model=proposed[b+j:b+j+1] if z-a==y-b else []) if changed else None)
                chosen.append((word['text'],word['bbox'],note))
            if changed and a==z:changes.append((len(chosen),record,None))
            continue
        if kind=='equal':
            for token,index in zip(proposed[b:y],range(a,z)):
                own=[word for word in witness if _same_occurrence(source[index],word)]
                word=own[0] if len(own)==1 else None
                second=([other for other in corroboration if _same_occurrence(word,other)
                    and _key(word['text'])==_key(other['text'])] if word else [])
                from . import text_quality
                lexical=lambda value:''.join(c for c in _key(value).casefold() if c.isalpha())
                vocabulary=text_quality.english_lexicon()
                lexical_repair=(lexical(token) not in vocabulary and
                    not any(c.isupper() for c in token) and word is not None and
                    lexical(word['text']) in vocabulary)
                corrected=bool(lexical_repair and word and word.get('confidence',0)>=90 and len(second)==1
                    and _key(word['text'])!=_key(token)
                    and _key(word['text']).rstrip('.,;:!?')!=_key(token).rstrip('.,;:!?')
                    and not _accent_conflict(word['text'],token))
                if corrected:
                    record=dict(source=[token],model=[token],chosen=[word['text']],status='RESOLVED',
                        method='two_own_pixel_readings',source_range=[index,index+1])
                    decisions.append(record);chosen.append((word['text'],word['bbox'],record))
                else:chosen.append((token,source[index]['bbox'],None))
            continue
        if z-a==y-b==1:
            source_word,model_word=original[a],candidate[b]
            stem=source_word.rstrip('.,;:!?')
            if stem and stem==model_word.rstrip('.,;:!?') and stem in (source_word,model_word):
                # Missing terminal punctuation is not a changed lexical word.
                # Preserve the first reading without manufacturing endpoint
                # proof from a source token that does not print that suffix.
                if unresolved_source and source_word!=model_word:
                    record=dict(source=[source_tokens[a]],model=[proposed[b]],chosen=[source_tokens[a]],
                        status='UNRESOLVED',method='retained_source_terminal_punctuation',
                        source_range=[a,z],model_range=[b,y])
                    decisions.append(record);chosen.append((source_tokens[a],source[a]['bbox'],record))
                else:chosen.append((proposed[b],source[a]['bbox'],None))
                continue
        alternatives=[('source',original[a:z],source_tokens[a:z]),
                      ('model',candidate[b:y],proposed[b:y])]
        # At least one nonempty adjacent anchor must locate the edit. Empty
        # alternatives need both; they cannot be proved at a cropped edge.
        left=original[max(0,a-2):a];right=original[z:z+2]
        supported=[]
        direct=[];boundary=None
        if z-a==y-b==1 and not any(len(_matches(pixels,left+seq+right))>1
                                   for name,seq,literal in alternatives):
            own=[w for w in witness if _same_occurrence(source[a],w)]
            if len(own)==1 and own[0].get('confidence',0)>=90:
                word=own[0];key=_key(word['text'])
                second=[w for w in corroboration if _same_occurrence(word,w) and _key(w['text'])==key]
                nearby=source[max(0,a-2):a]+source[z:z+2]
                anchored=any(len([w for w in witness if _same_occurrence(n,w)
                    and _key(n['text'])==_key(w['text']) and w.get('confidence',0)>=85])==1 for n in nearby)
                alternative=key in (original[a],candidate[b])
                similar=max(SequenceMatcher(None,placement(key),placement(t)).ratio()
                            for t in (original[a],candidate[b]))>=.65
                if (anchored or len(second)==1) and (alternative or len(second)==1 and similar):
                    direct=[('pixels',[word['text']],[word])]
        if 1<z-a<=3 and y-b==1 and ''.join(placement(t) for t in original[a:z])==placement(candidate[b]):
            owned=[[w for w in witness if _same_occurrence(s,w)] for s in source[a:z]]
            if all(len(row)==1 for row in owned):
                evidence=[row[0] for row in owned]
                distinct=len({id(w) for w in evidence})==len(evidence)
                literal=all(placement(w['text'])==placement(s['text'])
                            and w.get('confidence',0)>=90 for s,w in zip(source[a:z],evidence))
                neighbors=source[max(0,a-2):a]+source[z:z+2]
                anchored=any(len([w for w in witness if _same_occurrence(n,w)
                    and _key(n['text'])==_key(w['text']) and w.get('confidence',0)>=85])==1 for n in neighbors)
                if distinct and literal and anchored:
                    parts=_boundary_parts(proposed[b],source_tokens[a:z])
                    boundary=('boundary',parts,evidence)
                    if parts==source_tokens[a:z] and all(_key(s['text'])==_key(w['text']) for s,w in zip(source[a:z],evidence)):
                        direct=[('source',parts,evidence)]
        for name,seq,literal in ([] if direct else alternatives):
            if not (left or right) or not seq and not (left and right):continue
            locations=_matches(pixels,left+seq+right)
            if len(locations)!=1:continue
            context=witness[locations[0]:locations[0]+len(left+seq+right)]
            anchors=list(zip(source[max(0,a-2):a],context[:len(left)]))
            if right:anchors+=list(zip(source[z:z+2],context[-len(right):]))
            if not anchors or not all(_same_occurrence(a,b) for a,b in anchors):continue
            if not seq and (absence_proof is None or not absence_proof(source[a-1],source[z])):continue
            start=locations[0]+len(left)
            evidence=witness[start:start+len(seq)]
            if any(w.get('confidence',0)<85 for w in witness[locations[0]:locations[0]+len(left+seq+right)]):continue
            supported.append((name,literal,evidence))
        supported=direct or supported
        # A changed accent is not certified by an English recognizer's missing
        # or different mark. Preserve the first reader's codepoints and disclose
        # the unresolved choice, even when the bare letter sequence agrees.
        supported=[choice for choice in supported if not _accent_conflict(
            ''.join(choice[1]),''.join(proposed[b:y]))]
        # A row transition alone does not print a division. Preserve each
        # independently witnessed word boundary unless an actual terminal
        # hyphen in the original source proves the joined spelling.
        wrapped=(z-a>1 and y-b==1 and
                 any(t.endswith(('-', '\u00ad')) for t in source_tokens[a:z-1])
                 and ''.join(placement(t) for t in source_tokens[a:z])==placement(proposed[b]))
        # A crop that ends at a divided atom has not observed the complete
        # lexical word. Matching that prefix cannot certify deleting the
        # continuation from a complete first reading.
        partial_wrap=(z-a==y-b==1 and source_tokens[a].endswith(('-', '\u00ad'))
            and placement(proposed[b]).startswith(placement(source_tokens[a]))
            and len(placement(proposed[b]))>len(placement(source_tokens[a])))
        if wrapped or partial_wrap:supported=[];boundary=None
        # Letter noise resembling a number is not a verified replacement of a
        # well-formed ordinal. Inspect that exact original word with separate
        # language networks; if they cannot settle it, publish original ink.
        ordinal_conflict=(z-a==y-b==1 and _ordinal(proposed[b]) and any(
            len(choice[1])==1 and not _ordinal(choice[1][0]) and _ordinal_noise(choice[1][0])
            for choice in supported))
        if ordinal_conflict:supported=[]
        alternate=None;resampled=None;word_choice=None
        if len(supported)!=1 and z-a==y-b==1 and not partial_wrap:
            word_choice=_own_word_choice(source[a],witness)
            if word_choice and _accent_conflict(word_choice['text'],proposed[b]):word_choice=None
            if word_choice and max(SequenceMatcher(None,placement(word_choice['text']),placement(t)).ratio()
                                   for t in (source_tokens[a],proposed[b]))<.65:word_choice=None
            if word_choice:supported=[('word',[word_choice['text']],[word_choice])]
            else:resampled=_resampled_model_choice(source,a,witness,proposed[b])
            if resampled:supported=[('resampled',[resampled['text']],[resampled])]
            if not word_choice and not resampled:
                alternate=_alternate_choice(source,a,witness,proposed[b])
                if alternate:supported=[('alternate',[alternate['text']],[alternate])]
        if ordinal_conflict:supported=[choice for choice in supported if len(choice[1])==1 and _ordinal(choice[1][0])]
        resolved=len(supported)==1
        name,literal,evidence=(supported[0] if resolved else
            boundary if boundary else ('source',source_tokens[a:z],[]) if unresolved_source or not proposed[b:y] else ('model',proposed[b:y],[]))
        record=dict(source=source_tokens[a:z],model=proposed[b:y],chosen=list(literal),
                    status='RESOLVED' if resolved else 'UNRESOLVED',
                    method=('two_own_word_language_readings' if word_choice else 'corroborated_resampled_pixel_readings' if resampled else 'two_alternate_alphabet_pixel_readings' if alternate else 'independent_upright_crop') if resolved else 'pixel_word_boundary_only' if boundary else 'pixel_read_inconclusive',
                    source_range=[a,z],model_range=[b,y])
        if ordinal_conflict and not resolved:
            record.update(source_image_only=True,source_bbox=list(source[a]['bbox']))
        decisions.append(record)
        boxes=([w['bbox'] for w in evidence] if evidence else
               [w['bbox'] for w in source[a:z]])
        box=( [min(x[0] for x in boxes),min(x[1] for x in boxes),
               max(x[2] for x in boxes),max(x[3] for x in boxes)] if boxes else None)
        for j,token in enumerate(literal):
            word_record=_word_note(record,j,len(decisions)-1)
            chosen.append((token,evidence[j]['bbox'] if evidence else box,word_record))
        # A deletion still needs a control attached to an adjacent live word.
        if not literal:changes.append((len(chosen),record,box))
    # Reinsert source-proven note placeholders at their original model boundary.
    # Lexical edits never manufacture, delete or renumber a note reference.
    tokens=[t for t,_,_ in chosen];bounds={i:box for i,(_,box,_) in enumerate(chosen) if box}
    marks={i for i,(_,_,record) in enumerate(chosen) if record}
    notes={i:record for i,(_,_,record) in enumerate(chosen) if record}
    for at,record,box in changes:
        if not tokens:continue
        index=min(at,len(tokens)-1);marks.add(index);notes[index]=record
        if box:bounds[index]=box
    # Uncertainty positions are translated through the model/chosen alignment.
    translation=SequenceMatcher(None,proposed,tokens,autojunk=False)
    verified=[]
    for kind,a,z,b,y in translation.get_opcodes():
        if source_print:continue
        affected=([i for j,i in zip(range(a,z),range(b,y)) if visible[j][0] in pending]
            if kind=='equal' else list(range(b,y)) if any(visible[j][0] in pending for j in range(a,z)) else [])
        for i in affected:
            if proved_native_occurrence(tokens[i],bounds.get(i),native_print_words):continue
            # A reader's low-confidence flag is not permanent uncertainty when
            # the same occurrence has an exact, uniquely contextual pixel read.
            left=max(0,i-1);right=min(len(tokens),i+2)
            locations=_matches(pixels,[_key(t) for t in tokens[left:right]])
            context=witness[locations[0]:locations[0]+right-left] if len(locations)==1 else []
            own=[word for word in witness if i in bounds and
                 _complete_word_occurrence(dict(bbox=bounds[i]),word)]
            own_exact=(len(own)==1 and own[0].get('confidence',0)>=90
                       and _key(own[0]['text'])==_key(tokens[i]))
            source_own=[word for word in source if i in bounds and
                        _complete_word_occurrence(dict(bbox=bounds[i]),word)]
            source_exact=(len(source_own)==1 and source_own[0].get('confidence',0)>=90
                          and _key(source_own[0]['text'])==_key(tokens[i])
                          and not any(w.get('confidence',0)>=85 and _key(w['text'])!=_key(tokens[i])for w in own))
            if i not in notes and (own_exact or source_exact):
                verified.append(i);continue
            if i not in notes and context and all(w.get('confidence',0)>=90 for w in context) and all(
                    j in bounds and _same_occurrence(dict(bbox=bounds[j]),word)
                    for j,word in zip(range(left,right),context)):
                verified.append(i);continue
            marks.add(i)
            notes.setdefault(i,dict(source=[tokens[i]],model=[tokens[i]],chosen=[tokens[i]],
                status='UNRESOLVED',method='reader_uncertainty'))
    consumed=0;inserted=0
    for i,token in enumerate(model):
        if token.startswith('<n:'):
            at=0
            for _,a,z,b,y in translation.get_opcodes():
                if a<=consumed<=z:
                    at=b+min(y-b,max(0,consumed-a));break
            at+=inserted;inserted+=1
            tokens.insert(at,token)
            bounds={j+(j>=at):v for j,v in bounds.items()}
            marks={j+(j>=at) for j in marks};notes={j+(j>=at):v for j,v in notes.items()}
        else:consumed+=1
    return dict(tokens=tokens,boxes=bounds,marked=marks,decisions=decisions,word_notes=notes,verified_unchanged=verified)


def _engine_identity(engine):
    """Bind cached witnesses to the recognizer and installed language data."""
    return _cached_engine_identity(engine)


from functools import lru_cache
@lru_cache(maxsize=4)
def _cached_engine_identity(engine):
    import hashlib
    import os
    import re
    import subprocess
    from pathlib import Path
    version=subprocess.run([engine,'--version'],capture_output=True,timeout=5,check=True)
    languages=subprocess.run([engine,'--list-langs'],capture_output=True,timeout=5,check=True)
    digest=hashlib.sha256(version.stdout+version.stderr)
    listing=(languages.stdout+languages.stderr).decode('utf-8',errors='replace')
    roots=[os.environ.get('TESSDATA_PREFIX','')]
    roots.extend(re.findall(r'"([^"\n]+)"',listing))
    found=False
    for root in roots:
        if not root:continue
        for candidate in (Path(root)/'eng.traineddata',Path(root)/'tessdata/eng.traineddata'):
            if candidate.is_file():
                digest.update(candidate.read_bytes());found=True;break
        if found:break
    if not found:raise RuntimeError('Cannot fingerprint independent OCR language data')
    return digest.digest()


class CropWitness(list):
    """An unavailable second recognition does not invalidate other evidence."""
    failure = None


def read_crop(display, regions, cache):
    import subprocess
    try:
        result=CropWitness(_read_crop(display, regions, cache))
        # Recognition remains local and lazy. Ordinary settled English edits
        # incur no extra passes. The larger alphabet is used only to adjudicate
        # an unresolved single-word edit at its own physical occurrence.
        from functools import lru_cache
        @lru_cache(maxsize=1)
        def alternatives():
            try:
                import shutil
                engine=shutil.which('tesseract')
                directory=_latin_directory(engine) if engine else None
                if not directory:return []
                heights=[r[3]-r[1]for r in regions if r[3]>r[1]]
                # Tiny printed letters need enough samples to distinguish their
                # strokes. Both frames remain bounded by four million pixels.
                base=min(8,max(4,24/min(heights))) if heights else 4
                frames=[regions]
                contextual=alternate_regions(regions)
                if contextual!=[list(r)for r in regions]:frames.append(contextual)
                return [_read_crop(display,frame,cache,language='LatinRaw',scale=scale,tessdata=directory)
                        for frame in frames for scale in (base,base*1.25)]
            except (RuntimeError,OSError,ValueError,KeyError,subprocess.SubprocessError):return []
        @lru_cache(maxsize=1)
        def resampled():
            # Upscaling a small raster can reduce recognition confidence.
            # Two smaller frames may corroborate the model's own spelling;
            # neither frame alone can certify it or introduce another word.
            try:
                return [_read_crop(display,regions,cache,scale=scale) for scale in (1.5,2)]
            except (RuntimeError,OSError,ValueError,KeyError,subprocess.SubprocessError):return []
        result.resampled=resampled
        result.alternates=alternatives
        @lru_cache(maxsize=128)
        def own_word(bounds):
            try:
                import shutil
                engine=shutil.which('tesseract')
                directory=_latin_directory(engine) if engine else None
                if not directory:return []
                width,height=bounds[2]-bounds[0],bounds[3]-bounds[1]
                if not (0<width<=200 and 0<height<=80 and width/height<=18):return []
                return [_read_crop(display,[bounds],cache,language=language,scale=6,
                                  tessdata=data,psm=8)
                        for language,data in [('eng',None),('LatinRaw',directory)]]
            except (RuntimeError,OSError,ValueError,KeyError,subprocess.SubprocessError):return []
        result.word_readings=lambda word:own_word(tuple(word['bbox']))
        result.mark_absence=lambda word:below_dot_absent(display,word)
        return result
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result = CropWitness()
        result.failure = f'{type(error).__name__}: {error}'
        return result


def _read_crop(display, regions, cache, language='eng', scale=4, tessdata=None,psm=6):
    """Recognize upright source crops with bounded pixels and a separate layout.

    The full-page recognizer uses 3x/automatic page layout. This witness uses
    4x/single-block layout, cached by encoded pixels and recognition settings.
    A multi-region block is read in source-region order rather than flattening
    a bounding rectangle across unrelated columns.
    """
    import csv
    import fcntl
    import hashlib
    import io
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path
    engine=shutil.which('tesseract')
    if not engine:raise RuntimeError('Pixel adjudication requires local OCR')
    if not os.path.isfile(engine) or not os.access(engine,os.X_OK):
        raise RuntimeError('Independent OCR executable is unavailable')
    root=Path(cache);root.mkdir(parents=True,exist_ok=True);output=[]
    for region in regions:
        from . import extract
        box=extract.pymupdf.Rect(region)
        box.x0-=2;box.y0-=2;box.x1+=2;box.y1+=2;box &= display.rect
        pix=display.pixmap(box,scale=scale,max_pixels=4_000_000)
        image=pix.tobytes('png');border=0
        if language=='LatinRaw':
            from PIL import Image,ImageOps
            border=16
            picture=ImageOps.expand(Image.open(io.BytesIO(image)).convert('RGB'),border=border,fill='white')
            encoded=io.BytesIO();picture.save(encoded,format='PNG');image=encoded.getvalue()
        model_identity=(_latin_digest() if language=='LatinRaw' else b'')
        identity=hashlib.sha256(f'upright-block-{language}-psm{psm}-scale{scale}-v3\0'.encode()+model_identity+_engine_identity(engine)+image).hexdigest()
        target=root/(identity+'.json')
        with (root/(identity+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if target.exists():rows=json.loads(target.read_text())
            else:
                command=[engine,'stdin','stdout','-l',language,'--psm',str(psm)]
                if tessdata:command+=['--tessdata-dir',str(tessdata)]
                command+=['-c','tessedit_create_tsv=1']
                result=subprocess.run(command,
                    input=image,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45,
                    env=dict(os.environ,OMP_THREAD_LIMIT='1'))
                if result.returncode:raise RuntimeError('Independent crop recognition failed')
                rows=[]
                for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8')),
                    delimiter='\t',quoting=csv.QUOTE_NONE):
                    if row['level']!='5' or not row['text'].strip():continue
                    x,y,w,h=[int(row[k]) for k in ('left','top','width','height')]
                    rows.append(dict(text=row['text'],confidence=float(row['conf']),
                        bbox=[(x-border)/pix.width,(y-border)/pix.height,(x+w-border)/pix.width,(y+h-border)/pix.height]))
                temp=target.with_suffix('.'+str(os.getpid())+'.tmp')
                temp.write_text(json.dumps(rows,ensure_ascii=False));os.replace(temp,target)
        for row in rows:
            a,b,c,d=row['bbox']
            output.append(dict(row,bbox=[box.x0+a*box.width,box.y0+b*box.height,
                                         box.x0+c*box.width,box.y0+d*box.height]))
    return output


def refine_fallback(tokens, witness, lexicon, source=(), corroboration=()):
    """Resolve a damaged-layer fallback with positive, contextual pixel ink.

    Lexical support is secondary to a unique crop occurrence. It can support an
    OCR misspelling, but cannot rewrite an unfamiliar name into a common word or
    delete a real word simply because the dictionary prefers another sentence.
    """
    import re
    if any(t.startswith('<n:') for t in tokens) or not lexicon or not source:return None
    keys=[_key(t) for t in tokens];pixels=[_key(w['text']) for w in witness]
    matcher=SequenceMatcher(None,keys,pixels,autojunk=False)
    source_map={j:source[i] for a,b,n in SequenceMatcher(None,[_key(w['text']) for w in source],keys,autojunk=False).get_matching_blocks() for i,j in zip(range(a,a+n),range(b,b+n))}
    folded=SequenceMatcher(None,[t.casefold() for t in keys],
                           [t.casefold() for t in pixels],autojunk=False)
    if sum(n for _,_,n in folded.get_matching_blocks())/max(1,len(keys))<.75:return None
    known=lambda token:re.sub(r"[^\w'’]",'',token).casefold() in lexicon
    output=[];notes={};bounds={};decisions=[];gaps=[]
    for kind,a,z,b,y in matcher.get_opcodes():
        if kind=='equal':
            for i,j in zip(range(a,z),range(b,y)):
                if i in source_map:bounds[len(output)]=source_map[i]['bbox']
                output.append(tokens[i])
            continue
        left=keys[max(0,a-2):a];right=keys[z:z+2]
        evidence=witness[b:y]
        positive=bool(evidence) and all(w.get('confidence',0)>=90 for w in evidence)
        unique=len(_matches(pixels,left+pixels[b:y]+right))==1
        anchors=list(zip(range(max(0,a-2),a),witness[max(0,b-len(left)):b]))+list(zip(range(z,z+len(right)),witness[y:y+len(right)]))
        anchored=bool(left and right) and all(i in source_map and _same_occurrence(source_map[i],word) for i,word in anchors)
        lexical=(all(known(w['text']) for w in evidence) and
                 (not tokens[a:z] or
                  all(not any(c.isupper() for c in t) for t in tokens[a:z]) and
                  any(not known(t) for t in tokens[a:z])))
        numeral=(len(tokens[a:z])==len(evidence)==1 and
                 re.fullmatch(r'[IVXLCDM]+|[0-9]+',tokens[a]) and
                 re.fullmatch(r'[IVXLCDM]+|[0-9]+',evidence[0]['text']))
        owned_span=(len(tokens[a:z])==len(evidence) and bool(evidence) and all(
            i in source_map and _same_occurrence(source_map[i],word)
            for i,word in zip(range(a,z),evidence)))
        # Case changes preserve the lexical word. An unfamiliar spelling needs
        # two pixel readings at this occurrence, never a dictionary preference
        # or another copy elsewhere. Local OCR remains explicitly unverified.
        case_only=owned_span and all(_key(t).casefold()==_key(w['text']).casefold()
                                     for t,w in zip(tokens[a:z],evidence))
        two_views=owned_span and all(len([other for other in corroboration
            if _same_occurrence(word,other) and _key(other['text'])==_key(word['text'])])==1
            for word in evidence)
        resolved=bool(positive and ((unique and anchored and (lexical or numeral))
                                   or case_only or two_views))
        if _accent_conflict(''.join(tokens[a:z]),''.join(w['text'] for w in evidence)):resolved=False
        if kind=='insert' and not resolved and unique and anchored and a-1 in source_map and a in source_map:
            left_box=source_map[a-1]['bbox'];right_box=source_map[a]['bbox']
            height=max(left_box[3]-left_box[1],right_box[3]-right_box[1],1)
            # Additional recognized ink is not a proved letter. When it sits
            # inside this exact short same-row gap, retain the gap and following
            # word as original ink rather than silently dropping the edit.
            # A remote row or another column cannot own it.
            def in_gap(word):
                box=word['bbox'];small=min(box[3]-box[1],left_box[3]-left_box[1],right_box[3]-right_box[1])
                return (word.get('confidence',0)>=40 and any(c.isalnum()or c=='|'for c in word['text'])
                    and box[0]>=left_box[2]-.5 and box[2]<=right_box[0]+.5
                    and 0<right_box[0]-left_box[2]<=height*4
                    and min(box[3],left_box[3],right_box[3])-max(box[1],left_box[1],right_box[1])>=small*.5)
            if evidence and all(in_gap(word)for word in evidence):
                boxes=[right_box]+[word['bbox']for word in evidence]
                gaps.append((len(output),[min(b[0]for b in boxes),min(b[1]for b in boxes),
                    max(b[2]for b in boxes),max(b[3]for b in boxes)]))
        chosen=[w['text'] for w in evidence] if resolved else tokens[a:z]
        record=dict(source=tokens[a:z],model=tokens[a:z],chosen=chosen,
            status='RESOLVED' if resolved else 'UNRESOLVED',
            method=('two_local_pixel_readings' if two_views else 'pixel_case_agreement' if case_only
                    else 'contextual_crop_lexical_support') if resolved else 'pixel_read_inconclusive')
        decisions.append(record)
        for i,t in enumerate(chosen):
            index=len(output);output.append(t);notes[index]=_word_note(record,i,len(decisions)-1)
            if resolved:bounds[index]=evidence[i]['bbox']
            elif a+i in source_map:bounds[index]=source_map[a+i]['bbox']
    for index,box in gaps:
        if index>=len(output):continue
        record=dict(source=[output[index]],model=[output[index]],chosen=[output[index]],
            status='UNRESOLVED',method='unsettled_additional_pixel_ink',
            source_image_only=True,source_bbox=box)
        notes[index]=record;decisions.append(record)
    return dict(tokens=output,boxes=bounds,marked=set(notes),word_notes=notes,decisions=decisions)


def audit_agreement(tokens, source, witness, boundary_proof=None):
    """Flag pixel conflicts even where a layer and the first reading agree.

    A recognizer's different spelling is evidence of a conflict, not authority
    to replace a valid source word. Only the source occurrence's own overlapping
    ink can cause a mark; a different copy of a repeated word cannot do so.
    """
    visible=[(i,token)for i,token in enumerate(tokens) if not token.startswith('<n:')]
    if len(visible)!=len(source):return {}
    # Recognition may fuse adjacent printed words. Exact letters plus an
    # independently observed blank between every original occurrence can prove
    # the existing spaces. Geometry alone and a dictionary never waive a clash.
    segmented=set()
    if boundary_proof is not None:
        for other in witness:
            owned=[(j,w)for j,w in enumerate(source)if _same_occurrence(w,other)]
            if not 2<=len(owned)<=3 or other.get('confidence',0)<85:continue
            if ''.join(_key(w['text'])for j,w in owned)!=_key(other['text']):continue
            if all(boundary_proof(a,b)for (_,a),(_,b)in zip(owned,owned[1:])):
                segmented.update(j for j,w in owned)
    notes={}
    for j,((i,token),word) in enumerate(zip(visible,source)):
        if j in segmented:continue
        overlapping=[w for w in witness if _same_occurrence(word,w)]
        if len(overlapping)!=1:continue
        other=overlapping[0]
        if other.get('confidence',0)<85 or _key(other['text'])==_key(token):continue
        # Missing terminal punctuation is not a contradictory word reading.
        # Keep the original punctuation; letter, case and internal-symbol
        # differences still require occurrence-linked uncertainty.
        if _key(other['text']).rstrip('.,;:!?')==_key(token).rstrip('.,;:!?'):continue
        notes[i]=dict(source=[token],model=[token],chosen=[token],
            alternative=[other['text']],status='UNRESOLVED',method='agreeing_readings_pixel_conflict')
    return notes


def fragment_word_ranges(fragment, tokens):
    """Bind the complete authored stream across inline tags to raw ranges.

    A formatted word may enclose inline markup in its range. Callers retain that
    markup when adding a control and must not replace it as a plain text slice.
    Different inventories are refused; equal spelling never selects an owner.
    """
    import re
    from . import annotate
    plain=[];starts=[];ends=[];previous=None
    for node in annotate._text_nodes(fragment,include_uncertain=True):
        if previous is not None and re.search(r'</li\s*>',fragment[previous:node.raw_start]):
            plain.append(' ');starts.append(previous);ends.append(previous)
        plain.extend(node.plain);starts.extend(node.starts);ends.extend(node.ends)
        previous=node.ends[-1] if node.ends else previous
    words=[]
    for word in re.finditer(r'\S+',''.join(plain)):
        a,z=starts[word.start()],ends[word.end()-1]
        opened=[]
        for closing,name in re.findall(r'<(/?)(sup|sub|em|strong|i|b|a|span)\b[^>]*>',fragment[a:z],re.I):
            if closing:
                if opened and opened[-1]==name.casefold():opened.pop()
            else:opened.append(name.casefold())
        while opened:
            closing=re.match(r'</'+opened[-1]+r'\s*>',fragment[z:],re.I)
            if closing is None:break
            z+=closing.end();opened.pop()
        words.append((word.group(),a,z))
    if [word for word,_,_ in words]!=tokens:return None
    return [(a,z) for _,a,z in words]


def replace_formatted_word(fragment, chosen):
    """Replace text characters while preserving their inline style boundaries.

    Typed references, ordinals and other structural elements are deliberately
    outside this lexical edit. Character alignment maps existing style cuts;
    it never moves an action or strips its tags to obtain a spelling.
    """
    import re
    from html import escape
    from . import annotate
    if any(name.casefold() not in ('em','strong','i','b','span')
           for name in re.findall(r'</?([a-zA-Z][\w:-]*)\b',fragment)):
        return None
    nodes=[n for n in annotate._text_nodes(fragment,include_uncertain=True)if n.plain]
    original=''.join(n.plain for n in nodes)
    if not original or any(c.isspace()for c in original+chosen):return None
    def boundary(offset):
        if offset==len(original):return len(chosen)
        for _,a,z,b,y in SequenceMatcher(None,original,chosen,autojunk=False).get_opcodes():
            if a<=offset<z:return b+round((offset-a)*(y-b)/(z-a))
        return 0
    edits=[];offset=0
    for node in nodes:
        end=offset+len(node.plain)
        edits.append((node.starts[0],node.ends[-1],escape(chosen[boundary(offset):boundary(end)])))
        offset=end
    for a,z,value in reversed(edits):fragment=fragment[:a]+value+fragment[z:]
    return fragment



def _latin_digest():
    return bytes.fromhex('28acb38d9613595a7ae88a427bc7481c2ee488b960ea6baa31945f5ce0248b61')


def _latin_directory(engine):
    """Find only the hash-verified optical data installed at image build time."""
    import os
    return _cached_latin_directory(engine,os.environ.get('REFLOW_LATIN_TESSDATA',''),os.environ.get('TESSDATA_PREFIX',''))


@lru_cache(maxsize=8)
def _cached_latin_directory(engine,alternate_root,primary_root):
    import re,subprocess
    listing=subprocess.run([engine,'--list-langs'],capture_output=True,timeout=5,check=True)
    roots=[alternate_root,primary_root]
    roots+=re.findall(r'"([^"\n]+)"',(listing.stdout+listing.stderr).decode(errors='replace'))
    return _verified_latin_directory(tuple(roots))


@lru_cache(maxsize=8)
def _verified_latin_directory(roots):
    import hashlib
    from pathlib import Path
    for root in roots:
        if not root:continue
        for path in (Path(root)/'LatinRaw.traineddata',Path(root)/'tessdata/LatinRaw.traineddata'):
            if path.is_file() and hashlib.sha256(path.read_bytes()).digest()==_latin_digest():return path.parent
    return None


def _resampled_model_choice(source,index,witness,model_word):
    """Corroborate one model spelling without trusting an enlarged OCR guess.

    The primary crop must locate the same occurrence without a confident
    conflicting spelling. Its weak spelling need not repeat the model's.
    Two additional bounded frames must independently print it with confidence
    and a common neighboring anchor. English cannot establish accent changes.
    The caller retains the correction's source-linked mark after settlement.
    """
    reader=getattr(witness,'resampled',None)
    if reader is None:return None
    if any(c.isalpha() and not c.isascii() for c in source[index]['text']+model_word):return None
    own=[w for w in witness if _complete_word_occurrence(source[index],w)]
    if len(own)!=1:return None
    if (_key(own[0]['text'])!=_key(model_word)
            and own[0].get('confidence',0)>=85):return None
    choice=_alternate_pair_choice(source,index,witness,model_word,reader())
    if choice is None or _key(choice['text'])!=_key(model_word):return None
    return choice


def _alternate_choice(source,index,witness,model_word):
    reader=getattr(witness,'alternates',None)
    if reader is None:return None
    views=reader();choices={}
    for start in range(0,len(views),2):
        choice=_alternate_pair_choice(source,index,witness,model_word,views[start:start+2])
        if choice:choices[_key(choice['text'])]=choice
    # Two incompatible confident frame pairs cannot settle one printed word.
    return next(iter(choices.values()))if len(choices)==1 else None


def _alternate_pair_choice(source,index,witness,model_word,views):
    """Require two confident optical reads and a uniquely anchored occurrence.

    The script network includes macron vowels but no combining under-dot. Its
    omission cannot certify removing that mark from a first reading. English
    OCR is still never allowed to certify accent presence or absence.
    """
    if sum(c.isalnum()for c in source[index]['text'])<2:return None
    if len(views)!=2:return None
    own=[[w for w in view if _complete_word_occurrence(source[index],w)]for view in views]
    if not all(len(row)==1 and row[0].get('confidence',0)>=85 for row in own):return None
    first,second=own[0][0],own[1][0]
    if _key(first['text'])!=_key(second['text']):return None
    bare=lambda t:''.join(c for c in unicodedata.normalize('NFKD',t.casefold())if c.isalnum())
    if max(SequenceMatcher(None,bare(first['text']),bare(t)).ratio()
           for t in (source[index]['text'],model_word))<.65:return None
    marks=lambda value:{c for c in unicodedata.normalize('NFD',value)if unicodedata.combining(c)}
    if '\u0323'in marks(model_word)-marks(first['text']):
        proof=getattr(witness,'mark_absence',None)
        if proof is None or not proof(first):return None
    neighbors=source[max(0,index-2):index]+source[index+1:index+3]
    if not any(all(len([w for w in view if _same_occurrence(n,w)
            and _key(n['text']).casefold()==_key(w['text']).casefold()
            and w.get('confidence',0)>=85])==1 for view in views)for n in neighbors):return None
    return first


def disclose_alternate_conflicts(result, boxes, witness, allow_clear=True):
    """Retain the original ink for an already unresolved optical disagreement.

    A low-confidence alternate is evidence of ambiguity, not authority to print
    its guess. It can request a source-word image only at one physical word and
    only when neither alternate layout supports the live claim. Their own
    disagreement cannot select either spelling; it also requires original ink.
    """
    reader=getattr(witness,'alternates',None)
    if reader is None:return
    pending=[i for i in result['marked']if i in boxes
        and not result['tokens'][i].startswith('<n:')
        and (result.get('word_notes')or{}).get(i,{}).get('status')!='RESOLVED'
        and not (result.get('word_notes')or{}).get(i,{}).get('source_image_only')]
    if not pending:return
    views=reader()[:2]
    if len(views)!=2:return
    for index in pending:
        original=dict(bbox=boxes[index])
        own=[[w for w in view if _complete_word_occurrence(original,w)]for view in views]
        if not all(len(row)==1 for row in own):continue
        a,b=own[0][0],own[1][0]
        token=result['tokens'][index];note=(result.get('word_notes')or{}).get(index,{})
        if (allow_clear and all(_key(w['text'])==_key(token)and w.get('confidence',0)>=90 for w in (a,b))
                and note.get('source')==note.get('model')==note.get('chosen')==[token]):
            result['marked'].discard(index);result['word_notes'].pop(index)
            result.setdefault('verified_unchanged',[]).append(index)
            result['adjudication']=[d for d in result.get('adjudication',[])if d is not note]
            continue
        # A different local alphabet does not invalidate positive independent
        # evidence for this occurrence. Keep its original uncertainty, rather
        # than replacing supported prose with the alternate network's noise.
        own_primary=[w for w in witness if _complete_word_occurrence(original,w)]
        letters=lambda value:''.join(c for c in _key(value).casefold()if c.isalpha())
        lexical=letters(token)
        if (note.get('source')==note.get('chosen')==[token]
                and len(own_primary)==1 and lexical
                and letters(own_primary[0]['text'])==lexical):continue
        original_words=[w for w in getattr(witness,'source_occurrences',())
                        if _complete_word_occurrence(original,w)]
        if (len(original_words)==1 and original_words[0].get('confidence',0)>=85
                and lexical and letters(original_words[0]['text'])==lexical):
            # The original occurrence still supplies a supported alternative.
            # Do not resolve it: retain its marked, source-linked uncertainty.
            continue
        # Case, punctuation and raised-number suffixes have separate source
        # controls. Their disagreement cannot remove an unchanged lexical word.
        if lexical and any(letters(w['text'])==lexical for w in (a,b)):continue
        # An interpolated span box cannot authorize a particular word image.
        # Keep the existing unplaced source control rather than guessing an
        # occurrence inside that span from the alternate OCR output.
        if note.get('mapping')=='unplaced_within_span':continue
        if (len(own_primary)==1 and token.endswith(('-', '\u00ad', '\u2010', '\u2011'))
                and lexical and letters(own_primary[0]['text'])==lexical):continue
        source=note.get('source')or[]
        if (len(source)==1 and source[0].endswith(('-', '\u00ad', '\u2010', '\u2011'))
                and lexical.startswith(letters(source[0]))
                and all(_key(w['text']).strip()==_key(source[0]).strip()for w in (a,b))):
            # This crop owns only the printed prefix. It cannot contradict the
            # complete word whose physical continuation is checked separately.
            continue
        if (any(c.isalpha()for c in token) and any(c.isdigit()for c in token)
                and note.get('source')==note.get('model')==note.get('chosen')==[token]):
            # A letter/number identifier is not a guessed prose spelling. Keep
            # its literal instance and unresolved source control unless an
            # earlier positive adjudication actually proved the replacement.
            continue
        if any(_key(w['text'])==_key(token)for w in (a,b)):continue
        literal=lambda t:''.join(c for c in unicodedata.normalize('NFKD',t.casefold())if c.isalnum())
        if any(SequenceMatcher(None,literal(w['text']),literal(result['tokens'][index])).ratio()<.65 for w in (a,b)):continue
        note=(result.setdefault('word_notes',{}).get(index)or dict(source=[result['tokens'][index]],
            model=[result['tokens'][index]],chosen=[result['tokens'][index]],status='UNRESOLVED'))
        note=dict(note,source_image_only=True,source_bbox=a['bbox'],method='unsettled_alternate_pixel_readings')
        result['word_notes'][index]=note


def below_dot_absent(display,word):
    """Prove a missing below-baseline mark from untouched complete source ink.

    This is restricted to alphabetic words with no descenders. Multiple upright
    letter components must establish one baseline, and the entire lower strip
    must be blank. Cropped bottoms, punctuation, noise and ambiguous baselines
    all decline the proof rather than guessing a diacritic.
    """
    from . import extract,vision_pixels
    import statistics
    literal=''.join(c for c in unicodedata.normalize('NFD',word['text'])if not unicodedata.combining(c))
    if any(c.casefold()in 'gjpqy'for c in literal) or any(not(c.isalpha()or c=='-')for c in literal):return False
    a=word['bbox'];height=a[3]-a[1]
    bounds=extract.pymupdf.Rect(a[0]-.5,a[1]-.1*height,a[2]+.5,a[3]+.5*height)
    if any(bounds[i]<display.rect[i]for i in (0,1))or any(bounds[i]>display.rect[i]for i in (2,3)):return False
    pix=display.pixmap(bounds,scale=8,max_pixels=300_000)
    picture,components=vision_pixels._word_ink_components(pix)
    if not components:return False
    tallest=max(c[3]-c[1]for c in components)
    tall=[c for c in components if c[3]-c[1]>=tallest*.5]
    if len(tall)<3:return False
    baseline=statistics.median(c[3]for c in tall)
    if max(abs(c[3]-baseline)for c in tall)>max(2,tallest*.15):return False
    start=int(baseline+max(1,tallest*.06))
    if picture.height-start<tallest*.25:return False
    return picture.crop((0,start,picture.width,picture.height)).getextrema()[0]>=180


def alternate_regions(regions):
    """Give coherent consecutive rows context without joining other columns.

    These are recognition frames only. Every resulting decision still needs its
    own original word geometry; a rectangle never supplies lexical ownership.
    """
    result=[];previous=None
    for region in regions:
        box=list(region)
        if previous is not None:
            height=max(previous[3]-previous[1],box[3]-box[1])
            overlap=min(previous[2],box[2])-max(previous[0],box[0])
            same_column=overlap>=.8*min(previous[2]-previous[0],box[2]-box[0])
            consecutive=0<=box[1]-previous[3]<=height*1.5
            if same_column and consecutive:
                merged=result[-1];result[-1]=[min(merged[0],box[0]),min(merged[1],box[1]),max(merged[2],box[2]),max(merged[3],box[3])]
            else:result.append(box)
        else:result.append(box)
        previous=box
    return result



def _complete_word_occurrence(source,word):
    """An internal punctuation atom cannot own a larger recognized word."""
    if not _same_occurrence(source,word):return False
    a,b=source['bbox'],word['bbox']
    if source.get('native_parts') and not all(
            b[0]-.2<=(p['bbox'][0]+p['bbox'][2])/2<=b[2]+.2 and
            b[1]-.2<=(p['bbox'][1]+p['bbox'][3])/2<=b[3]+.2
            for p in source['native_parts']):return False
    return min(a[2],b[2])-max(a[0],b[0])>=.6*(b[2]-b[0])

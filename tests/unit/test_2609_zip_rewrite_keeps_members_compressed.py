# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
"""Replacing one member of a book archive must not re-deflate the others (#2609).

Kobo downloads with "Embed metadata" on rebuild the KEPUB before the first byte
is sent. Rebuilding by decompressing and recompressing every member made the
time to first byte grow with the book: a ~1 GB image-heavy KEPUB took longer
than Nickel waits, so the download never finished. Only ``content.opf`` (or the
cover, for a personal-cover copy) changes, so every other member's compressed
bytes can go into the new archive as they are.

The observable proof that a member was not recompressed is that its compressed
bytes in the new archive are identical to the source's. The fixtures compress
text at level 1, so a re-deflate at the default level produces different bytes.
"""
from __future__ import annotations

import io
import os
import random
import struct
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from PIL import Image

from cps.epub_helper import updateEpub

pytestmark = pytest.mark.unit

CONTAINER = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

PACKAGE = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata><meta name="cover" content="cover-image"/></metadata>
  <manifest>
    <item id="cover-image" href="images/cover.jpg" media-type="image/jpeg"/>
    <item id="chapter" href="text/ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest><spine><itemref idref="chapter"/></spine>
</package>"""

NEW_PACKAGE = PACKAGE.replace(b"<metadata>", b"<metadata><dc:title xmlns:dc=\"http://purl.org/dc/elements/1.1/\">New</dc:title>")


def _jpeg(color):
    stream = io.BytesIO()
    Image.new("RGB", (12, 18), color).save(stream, "JPEG")
    return stream.getvalue()


def _compressible_text(seed, size):
    rng = random.Random(seed)
    words = [b"alpha", b"bravo", b"charlie", b"delta", b"echo", b"foxtrot", b"golf"]
    out = bytearray(b"<html><body><p>")
    while len(out) < size:
        out += rng.choice(words) + b" "
    return bytes(out) + b"</p></body></html>"


class _Unseekable(io.RawIOBase):
    """A write-only stream, so zipfile writes data descriptors (flag bit 3)."""

    def __init__(self):
        self.buffer = io.BytesIO()

    def writable(self):
        return True

    def write(self, data):
        return self.buffer.write(data)

    def tell(self):
        return self.buffer.tell()

    def seekable(self):
        return False

    def seek(self, *_args):
        raise io.UnsupportedOperation("seek")


def _write_book(archive, cover=None):
    archive.comment = b"source archive comment"
    archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
    archive.writestr("META-INF/container.xml", CONTAINER)
    archive.writestr("OEBPS/content.opf", PACKAGE)
    archive.writestr("OEBPS/text/ch1.xhtml", _compressible_text(1, 200_000))
    archive.writestr("OEBPS/text/ch2.xhtml", _compressible_text(2, 120_000))
    archive.writestr("OEBPS/images/cover.jpg", cover or _jpeg("blue"), compress_type=zipfile.ZIP_STORED)
    archive.writestr("OEBPS/images/noise.bin", random.Random(3).randbytes(150_000),
                     compress_type=zipfile.ZIP_STORED)
    zip64_member = zipfile.ZipInfo("OEBPS/text/zip64.xhtml", (2020, 1, 2, 3, 4, 6))
    zip64_member.compress_type = zipfile.ZIP_DEFLATED
    zip64_member._compresslevel = 1  # ZipFile.open ignores the archive's level for a ZipInfo
    with archive.open(zip64_member, "w", force_zip64=True) as member:
        member.write(_compressible_text(4, 50_000))


def _seekable_book(path, cover=None):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        _write_book(archive, cover)


def _streamed_book(path):
    stream = _Unseekable()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        _write_book(archive)
    with open(path, "wb") as handle:
        handle.write(stream.buffer.getvalue())


def _raw_member(path, info):
    """The member's compressed bytes, read from its local file header."""
    with open(path, "rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        signature, *_rest, name_len, extra_len = struct.unpack("<4s5H3L2H", header)
        assert signature == b"PK\x03\x04"
        handle.seek(info.header_offset + 30 + name_len + extra_len)
        return handle.read(info.compress_size)


def _assert_only_member_replaced(source, dest, replaced, new_bytes):
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(dest) as out:
        assert out.testzip() is None
        names = out.namelist()
        assert len(names) == len(set(names)), "a member was written twice"
        assert sorted(names) == sorted(src.namelist())
        assert out.infolist()[0].filename == "mimetype"
        assert out.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert out.comment == src.comment
        assert out.read(replaced) == new_bytes

        not_recompressed = []
        for info in src.infolist():
            if info.filename == replaced:
                continue
            copied = out.getinfo(info.filename)
            assert out.read(info.filename) == src.read(info.filename), info.filename
            assert copied.compress_type == info.compress_type, info.filename
            assert copied.CRC == info.CRC, info.filename
            assert copied.date_time == info.date_time, info.filename
            if _raw_member(dest, copied) != _raw_member(source, info):
                not_recompressed.append(info.filename)
        assert not_recompressed == [], "members were re-deflated instead of copied: %s" % not_recompressed


def test_metadata_embed_copies_every_other_member_compressed_bytes_verbatim(tmp_path):
    source = tmp_path / "book.kepub"
    dest = tmp_path / "out.kepub"
    _seekable_book(source)

    updateEpub(str(source), str(dest), "OEBPS/content.opf", NEW_PACKAGE)

    _assert_only_member_replaced(source, dest, "OEBPS/content.opf", NEW_PACKAGE)


def test_metadata_embed_handles_a_source_written_with_data_descriptors(tmp_path):
    source = tmp_path / "streamed.kepub"
    dest = tmp_path / "out.kepub"
    _streamed_book(source)
    with zipfile.ZipFile(source) as src:
        assert all(info.flag_bits & 0x08 for info in src.infolist()), "fixture lost its data descriptors"

    updateEpub(str(source), str(dest), "OEBPS/content.opf", NEW_PACKAGE)

    _assert_only_member_replaced(source, dest, "OEBPS/content.opf", NEW_PACKAGE)


def test_personal_cover_copy_copies_every_other_member_compressed_bytes_verbatim(tmp_path, monkeypatch):
    from cps import helper
    from cps.services import user_cover

    monkeypatch.setattr(user_cover.constants, "CONFIG_DIR", str(tmp_path / "config"))
    source = tmp_path / "library.epub"
    _seekable_book(source, cover=_jpeg("blue"))
    row = SimpleNamespace(user_id=7, book_id=11, updated_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    os.makedirs(user_cover.cover_directory(7), exist_ok=True)
    with open(user_cover.path_for_row(row), "wb") as cover_file:
        cover_file.write(_jpeg("red"))
    monkeypatch.setattr(
        user_cover, "override_for_user",
        lambda user_id, book_id, session=None: row if (user_id, book_id) == (7, 11) else None,
    )
    monkeypatch.setattr(helper, "get_temp_dir", lambda: str(tmp_path / "deliveries"))

    delivered = user_cover.materialize_delivery_copy(7, 11, str(source), "epub")

    assert delivered is not None
    dest = os.path.join(delivered[0], delivered[1] + ".epub")
    with zipfile.ZipFile(dest) as out:
        new_cover = out.read("OEBPS/images/cover.jpg")
    assert new_cover != _jpeg("blue")
    _assert_only_member_replaced(source, dest, "OEBPS/images/cover.jpg", new_cover)

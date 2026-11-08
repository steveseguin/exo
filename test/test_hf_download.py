import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
from tenacity import stop_after_attempt, wait_none

from exo.download.hf import hf_helpers


class Response:
  def __init__(self, status, headers, body=b""):
    self.status = status
    self.headers = headers
    self.body = body
    self.content = self
    self.request_info = SimpleNamespace(real_url="https://example.invalid/model")
    self.history = ()

  async def __aenter__(self):
    return self

  async def __aexit__(self, *args):
    pass

  async def iter_chunked(self, size):
    for offset in range(0, len(self.body), size):
      yield self.body[offset:offset + size]


class Session:
  def __init__(self, responses):
    self.responses = iter(responses)
    self.requests = []

  def get(self, url, headers):
    self.requests.append(dict(headers))
    return next(self.responses)


class TestDownloadFile(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.path = Path(self.directory.name)/"model.safetensors"
    self.download = hf_helpers.download_file.retry_with(stop=stop_after_attempt(1), wait=wait_none())
    for name, value in (
      ("get_auth_headers", AsyncMock(side_effect=lambda: {})),
      ("get_file_download_percentage", AsyncMock(return_value=0)),
      ("download_file", self.download),
    ):
      context = patch.object(hf_helpers, name, value)
      context.start()
      self.addCleanup(context.stop)

  async def run_download(self, session, progress=None):
    await hf_helpers.download_file(session, "test/model", "main", self.path.name, self.directory.name, progress)

  async def test_oversized_file_restarts_without_appending_error_body(self):
    self.path.write_bytes(b"old-model-too-large")
    session = Session([
      Response(416, {"Content-Range": "bytes */5", "Content-Length": "9"}, b"bad range"),
      Response(200, {"Content-Length": "5"}, b"model"),
    ])
    progress = AsyncMock()
    await self.run_download(session, progress)
    self.assertEqual(self.path.read_bytes(), b"model")
    self.assertEqual(session.requests, [{"Range": "bytes=19-"}, {}])
    self.assertEqual(progress.call_args.args[0].downloaded, 5)
    self.assertEqual(progress.call_args.args[0].status, "complete")

  async def test_complete_file_is_not_downloaded_again(self):
    self.path.write_bytes(b"model")
    session = Session([Response(416, {"Content-Range": "bytes */5"}, b"bad range")])
    await self.run_download(session)
    self.assertEqual(self.path.read_bytes(), b"model")
    self.assertEqual(len(session.requests), 1)

  async def test_partial_content_resumes(self):
    self.path.write_bytes(b"mo")
    session = Session([Response(206, {"Content-Length": "3", "Content-Range": "bytes 2-4/5"}, b"del")])
    await self.run_download(session)
    self.assertEqual(self.path.read_bytes(), b"model")
    self.assertEqual(session.requests, [{"Range": "bytes=2-"}])

  async def test_ignored_range_replaces_existing_content(self):
    self.path.write_bytes(b"old")
    session = Session([Response(200, {"Content-Length": "5"}, b"model")])
    await self.run_download(session)
    self.assertEqual(self.path.read_bytes(), b"model")

  async def test_malformed_range_restarts(self):
    self.path.write_bytes(b"old-model-too-large")
    session = Session([
      Response(416, {"Content-Range": "invalid"}, b"bad range"),
      Response(200, {"Content-Length": "5"}, b"model"),
    ])
    await self.run_download(session)
    self.assertEqual(self.path.read_bytes(), b"model")
    self.assertEqual(session.requests[-1], {})

  async def test_rejected_full_request_raises_without_modifying_file(self):
    self.path.write_bytes(b"old-model-too-large")
    session = Session([
      Response(416, {"Content-Range": "bytes */5"}, b"bad range"),
      Response(416, {"Content-Range": "bytes */5"}, b"bad range"),
    ])
    with self.assertRaises(aiohttp.ClientResponseError):
      await self.run_download(session)
    self.assertEqual(self.path.read_bytes(), b"old-model-too-large")
    self.assertEqual(len(session.requests), 2)


if __name__ == "__main__":
  unittest.main()

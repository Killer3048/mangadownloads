from __future__ import annotations

import base64
import io
import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict

import requests
from PIL import Image


@dataclass(frozen=True)
class XaiImagineConfig:
  api_key: str
  base_url: str = "https://api.x.ai/v1"
  model: str = "grok-imagine-image"
  timeout_sec: int = 120
  # requests' "verify" parameter: bool or CA bundle path.
  tls_verify: bool | str = True


class XaiImagineError(RuntimeError):
  def __init__(self, message: str, status_code: int | None = None, retry_after_sec: float | None = None):
    super().__init__(message)
    self.status_code = status_code
    self.retry_after_sec = retry_after_sec


def _parse_int(value: Any, default: int) -> int:
  try:
    n = int(value)
    return n if n > 0 else default
  except Exception:
    return default


def _parse_bool(value: Any, default: bool) -> bool:
  if isinstance(value, bool):
    return value
  if value is None:
    return default
  if isinstance(value, (int, float)):
    return bool(value)
  s = str(value).strip().lower()
  if s in ("1", "true", "yes", "y", "on"):
    return True
  if s in ("0", "false", "no", "n", "off"):
    return False
  return default


def _resolve_tls_verify(raw: Dict[str, Any]) -> bool | str:
  # Allow env override (handy for quick troubleshooting).
  env = (os.environ.get("XAI_TLS_VERIFY") or "").strip()
  if env:
    if env.lower() == "certifi":
      try:
        import certifi  # type: ignore

        return certifi.where()
      except Exception:
        return True
    return _parse_bool(env, True)

  ca_bundle = raw.get("tls_ca_bundle") or raw.get("ca_bundle") or raw.get("ca_bundle_path")
  if isinstance(ca_bundle, str) and ca_bundle.strip():
    if ca_bundle.strip().lower() == "certifi":
      try:
        import certifi  # type: ignore

        return certifi.where()
      except Exception:
        return True
    return ca_bundle.strip()

  # Keep strict verification on by default.
  verify = raw.get("tls_verify")
  if verify is None:
    verify = raw.get("verify_tls")
  if verify is None:
    verify = raw.get("verify_ssl")
  if verify is None:
    verify = raw.get("ssl_verify")
  if verify is None:
    return True

  if isinstance(verify, str) and verify.strip().lower() == "certifi":
    try:
      import certifi  # type: ignore

      return certifi.where()
    except Exception:
      return True

  return _parse_bool(verify, True)


def load_xai_imagine_config(raw: Dict[str, Any], api_key: str) -> XaiImagineConfig:
  base_url = str(raw.get("base_url") or "https://api.x.ai/v1").strip()
  model = str(raw.get("model_image") or raw.get("model") or "grok-imagine-image").strip()
  timeout_sec = _parse_int(raw.get("timeout_sec"), 120)
  tls_verify = _resolve_tls_verify(raw)
  return XaiImagineConfig(api_key=api_key, base_url=base_url, model=model, timeout_sec=timeout_sec, tls_verify=tls_verify)


def _image_to_data_uri_png(img: Image.Image) -> str:
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  b64 = base64.b64encode(buf.getvalue()).decode("ascii")
  return f"data:image/png;base64,{b64}"


def _decode_image_bytes(data: bytes) -> Image.Image:
  try:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im
  except Exception as e:
    raise XaiImagineError(f"Failed to decode xAI image bytes: {e}") from e


def _format_xai_error(resp: requests.Response, data: Dict[str, Any]) -> str:
  """
  xAI error payloads are not fully stable (sometimes JSON, sometimes plain text).
  Keep this best-effort and concise.
  """
  try:
    if isinstance(data, dict) and data:
      code = data.get("code")
      err = data.get("error")
      if isinstance(code, str) and isinstance(err, str) and code.strip() and err.strip():
        return f"{code}: {err}"
      if isinstance(err, str) and err.strip():
        return err
      if isinstance(err, dict):
        msg = err.get("message") or err.get("detail") or err.get("error")
        if isinstance(msg, str) and msg.strip():
          return f"{code}: {msg}" if isinstance(code, str) and code.strip() else msg
      detail = data.get("detail")
      if isinstance(detail, str) and detail.strip():
        return detail
      if isinstance(detail, dict):
        msg = detail.get("message") or detail.get("error") or detail.get("detail")
        if isinstance(msg, str) and msg.strip():
          return msg
      msg = data.get("message")
      if isinstance(msg, str) and msg.strip():
        return msg
  except Exception:
    pass

  text = (resp.text or "").strip()
  return text or "Unknown xAI error."


def _format_ssl_error(exc: Exception, url: str, cfg: XaiImagineConfig) -> str:
  msg = str(exc) or exc.__class__.__name__
  now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
  lower = msg.lower()

  if "certificate verify failed" in lower and "certificate has expired" in lower:
    skew_hint = ""
    try:
      # This call is only for diagnostics. Use plain HTTP so it works even when local time is wrong.
      r = requests.head("http://x.ai", timeout=3, allow_redirects=False)
      remote_date = (r.headers.get("Date") or r.headers.get("date") or "").strip()
      if remote_date:
        remote_utc = parsedate_to_datetime(remote_date).astimezone(timezone.utc).replace(microsecond=0)
        delta = datetime.now(timezone.utc) - remote_utc
        if abs(delta) >= timedelta(hours=12):
          days = abs(delta.total_seconds()) / 86400.0
          direction = "ahead" if delta.total_seconds() > 0 else "behind"
          skew_hint = f" System clock seems ~{days:.1f} days {direction} (remote Date: {remote_utc.isoformat().replace('+00:00','Z')})."
    except Exception:
      skew_hint = ""
    return (
      "TLS certificate verification failed (certificate has expired). "
      f"Local UTC time: {now_utc}.{skew_hint} "
      "Check system date/time and CA certificates. "
      "Temporary workaround: set `xai.tls_verify: false` in keys.yaml (INSECURE). "
      f"Target: {url}"
    )

  if "certificate verify failed" in lower:
    return (
      "TLS certificate verification failed. "
      f"Local UTC time: {now_utc}. "
      "Check CA certificates (certifi/ca-certificates) or try `xai.tls_ca_bundle: certifi`. "
      f"Target: {url}. Detail: {msg}"
    )

  return f"TLS/SSL error while calling xAI. Target: {url}. Detail: {msg}"


def imagine_edit_image(
  img: Image.Image,
  prompt: str,
  cfg: XaiImagineConfig,
) -> Image.Image:
  """
  Call xAI image edit via REST.
  Uses /images/edits per xAI docs (image + prompt).
  NOTE: Observed API expects `image` as an object (ImageUrl), not a plain string.
  """
  if not cfg.api_key:
    raise XaiImagineError("Missing xAI API key.")

  data_uri = _image_to_data_uri_png(img)
  # Per xAI docs, image edit uses /images/edits (not /images/generations).
  url = cfg.base_url.rstrip("/") + "/images/edits"
  headers = {
    "Authorization": f"Bearer {cfg.api_key}",
    "Content-Type": "application/json",
  }
  payload = {
    "model": cfg.model,
    "prompt": prompt,
    # Verified against xAI API responses: REST edit expects `image` as an object (ImageUrl),
    # e.g. {"url": "data:image/png;base64,..."} (a plain string causes 422).
    "image": {"url": data_uri},
    "response_format": "b64_json",
  }

  try:
    resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_sec, verify=cfg.tls_verify)
  except requests.exceptions.SSLError as e:
    raise XaiImagineError(_format_ssl_error(e, url, cfg)) from e
  except Exception as e:
    raise XaiImagineError(f"xAI request failed: {e}") from e

  data: Dict[str, Any] = {}
  try:
    data = resp.json() or {}
  except Exception:
    data = {}

  if not resp.ok:
    detail = _format_xai_error(resp, data)
    retry_after_sec: float | None = None
    try:
      ra = resp.headers.get("Retry-After")
      if ra:
        retry_after_sec = float(str(ra).strip())
    except Exception:
      retry_after_sec = None
    raise XaiImagineError(
      f"xAI error {resp.status_code}: {detail}",
      status_code=int(resp.status_code),
      retry_after_sec=retry_after_sec,
    )

  items = data.get("data")
  if not isinstance(items, list) or not items:
    raise XaiImagineError("xAI response missing data[].")

  first = items[0] if isinstance(items[0], dict) else {}
  if not isinstance(first, dict):
    raise XaiImagineError("xAI response data[0] invalid.")

  b64_json = first.get("b64_json")
  if isinstance(b64_json, str) and b64_json.strip():
    try:
      raw = base64.b64decode(b64_json)
    except Exception as e:
      raise XaiImagineError(f"xAI response b64_json decode failed: {e}") from e
    return _decode_image_bytes(raw)

  url_out = first.get("url")
  if isinstance(url_out, str) and url_out.strip():
    try:
      r2 = requests.get(url_out, timeout=cfg.timeout_sec)
      if not r2.ok:
        raise XaiImagineError(f"xAI image download failed {r2.status_code}: {r2.text}")
      return _decode_image_bytes(r2.content)
    except XaiImagineError:
      raise
    except Exception as e:
      raise XaiImagineError(f"xAI image download failed: {e}") from e

  raise XaiImagineError("xAI response contains neither b64_json nor url.")

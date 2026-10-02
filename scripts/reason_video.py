"""Run Cosmos3 Nano locally and save reviewable ingredient-action evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = {"picked_up", "used_in_preparation", "returned", "discarded", "served", "unknown"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_evidence(text: str, duration: float | None = None) -> dict:
    # Some reasoning checkpoints emit a thinking block even when JSON is requested.
    final = text.rsplit("</think>", 1)[-1].strip()
    if final.startswith("```"):
        final = final.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    result = json.loads(final)
    if not isinstance(result, dict) or not isinstance(result.get("events"), list):
        raise ValueError("Model output must be an object containing an events list")
    events = []
    for event in result["events"]:
        if not isinstance(event, dict):
            raise ValueError("Each event must be an object")
        if event.get("action") not in ACTIONS:
            raise ValueError("Unrecognized action")
        for key in ("ingredient", "evidence", "uncertainty"):
            if not isinstance(event.get(key), str):
                raise ValueError(f"Event {key} must be text")
        if not event["ingredient"].strip() or not event["evidence"].strip():
            raise ValueError("Event ingredient and evidence cannot be empty")
        start, end = event.get("start_seconds"), event.get("end_seconds")
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) for t in (start, end)):
            raise ValueError("Event timestamps must be numbers")
        if not 0 <= start <= end or not all(math.isfinite(t) for t in (start, end)):
            raise ValueError("Event timestamps must be finite and ordered")
        if duration is not None and end > duration + 0.5:
            raise ValueError("Event timestamp exceeds the clip duration")
        events.append({key: event[key] for key in (
            "ingredient", "action", "start_seconds", "end_seconds", "evidence", "uncertainty",
        )})
    return {"events": events}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Cosmos3-Nano")
    parser.add_argument("--ingredient", action="append", default=[], help="Candidate ingredient; repeat as needed")
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evidence.json")
    args = parser.parse_args()
    video = args.video.resolve(strict=True)
    if not video.is_file() or not math.isfinite(args.fps) or args.fps <= 0 or args.max_new_tokens <= 0:
        parser.error("Provide a video file, positive FPS, and positive token limit")
    checkpoint = str(args.model.resolve(strict=True))
    weight_index = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())
    missing = [name for name in set(weight_index["weight_map"].values()) if not (Path(checkpoint) / name).is_file()]
    if missing:
        raise RuntimeError(f"Model download is incomplete ({len(missing)} missing weight files). Run bash scripts/download_cosmos.sh")

    import torch
    import av
    from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    with av.open(str(video)) as container:
        if not container.streams.video:
            parser.error("Input has no video stream")
        duration = container.duration / av.time_base if container.duration is not None else None
    revision_metadata = Path(checkpoint) / ".cache/huggingface/download/config.json.metadata"
    revision = revision_metadata.read_text().splitlines()[0] if revision_metadata.is_file() else None
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = Cosmos3OmniForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="auto", local_files_only=True,
    )
    candidates = json.dumps(args.ingredient)
    prompt = f"""Observe the kitchen video and identify ingredient handling events.
Candidate ingredient names: {candidates}. They are hints, not proof of what appears.
Use unknown when identity or action is unclear. Describe only visible evidence.
Distinguish picking something up, using it in preparation, returning it, discarding
it, and serving food. A returned container may have had some contents used earlier;
record each visible action separately. Do not infer a sale or quantities consumed
from visibility, container motion, or apparent portion size. Recipes and POS data
will be reconciled separately. Timestamps are seconds from the start of this clip.
Return only a JSON object, with no reasoning or markdown, in this shape:
{{"events": [{{"ingredient": "name or unknown", "action": "picked_up|used_in_preparation|returned|discarded|served|unknown", "start_seconds": 0.0, "end_seconds": 1.0, "evidence": "visible observation", "uncertainty": "ambiguity, or empty string"}}]}}
Use an empty events list when no ingredient action is visible."""
    messages = [{"role": "user", "content": [
        {"type": "video", "path": str(video)},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", processor_kwargs={"fps": args.fps},
    ).to(model.device, torch.bfloat16)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    response = processor.batch_decode(
        generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".raw.txt")
    raw_path.write_text(response, encoding="utf-8")
    try:
        evidence = parse_evidence(response, duration)
    except (ValueError, TypeError) as error:
        print(f"Invalid evidence: {error}. Raw response saved to {raw_path}", file=sys.stderr)
        return 1
    result = {
        "clip_sha256": file_sha256(video),
        "video": str(video),
        "model": "nvidia/Cosmos3-Nano",
        "model_revision": revision,
        "model_path": checkpoint,
        "fps": args.fps,
        "duration_seconds": duration,
        "status": "needs_review",
        **evidence,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

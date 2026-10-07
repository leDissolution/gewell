"""Read complete records from native ``--log-format json`` output."""

import json


def read_records(path):
    # A running process can be midway through its final write. Complete lines
    # must still parse: malformed diagnostics should fail their smoke test.
    return [json.loads(line) for line in path.read_text(errors="replace").split("\n")[:-1]]


def fields(path):
    return {record["name"]: record["value"] for record in read_records(path)
            if record["event"] == "field"}


def events(path):
    return [{"kind": record["event"], "value": record["data"]}
            for record in read_records(path) if "data" in record]

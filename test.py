import json, pathlib
# Inspect how many pilot records are valid vs failed
path = pathlib.Path('data/experiment/direction2_pilot/direction2_checkpoint.jsonl')
valid, failed = 0, 0
for line in path.open(encoding='utf-8'):
    r = json.loads(line)
    if r.get('parsed_choice') in ('A', 'B'):
        valid += 1
    elif r.get('error') is None:
        failed += 1
print(f'valid: {valid}, parse_failed: {failed}')
#!/usr/bin/env python3
"""Opt-in HTTP startup/streaming/logprob smoke for either native model bundle."""
import argparse
import json
from pathlib import Path
from online_http_smoke import native_server, wait_idle, require
from online_tools_smoke import collect, equivalent
from native_log import fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('build/gewell'))
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir()
    model = 'model-selection-test'
    report = {}
    with native_server(args.binary.resolve(), args.bundle.resolve(), work/'server', model, 2, 2048, 0, 120) as (client, port, log):
        for chat in (False, True):
            options = {'messages': [{'role':'user', 'content':'Say hello.'}]} if chat else {'prompt': [2,9259]}
            key = 'chat' if chat else 'raw'
            buffered = collect(client, model, max_tokens=8, **options)
            streamed = collect(client, model, max_tokens=8, stream=True, **options)
            equivalent(buffered, streamed)
            require(streamed['usage']['prompt_tokens_details']['cached_tokens'] > 0, 'repeat prompt missed cache')
            report[key] = {'buffered':buffered, 'streamed':streamed}
        result = client.chat.completions.create(model=model, messages=[{'role':'user','content':'Say hello.'}],
                                               temperature=0, max_tokens=4, logprobs=True, top_logprobs=20)
        payload = result.model_dump(exclude_unset=True)
        scores = payload['choices'][0]['logprobs']
        require(scores is not None and scores.get('content'), 'chat logprobs missing')
        require(''.join(entry['token'] for entry in scores['content']) == payload['choices'][0]['message']['content'],
                'logprob tokens disagree with emitted content')
        for entry in scores['content']:
            require(entry['logprob'] == 0 and len(entry['top_logprobs']) == 1 and
                    entry['top_logprobs'][0]['token'] == entry['token'] and
                    entry['top_logprobs'][0]['logprob'] == 0, 'incorrect greedy logprobs')
        report['logprobs'] = payload
        report['idle'] = wait_idle(port, 120)
        startup = fields(log)
        require(startup['server_image_limit'] == startup['server_image_prompt_tokens'] ==
                startup['server_image_max_soft_tokens'] == 0 and
                startup['server_image_admission'] == 'disabled' and
                startup['server_image_prefix_cache'] is False, 'disabled vision advertised as available')
        manifest = json.loads((args.bundle/'manifest.json').read_text())
        require(startup['payload_sha256'] == manifest['artifact']['payload_sha256'], 'artifact provenance differs')
        report['startup'] = startup
    (work/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('HTTP model selection: raw/chat buffered-SSE parity, cache reuse, logprobs, cleanup PASS')


if __name__ == '__main__':
    main()

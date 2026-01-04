from __future__ import annotations

from dotenv import load_dotenv

from smartagent.renderer import _DEFAULT_RENDERER
from smartagent.agent import build_agent


load_dotenv(".env", override=True)

agent = build_agent()

request_dict = {
    "report generation": "Write me a /final_report.md based on the files from the zip file inside the /workspace, write the summary report in pure Chinese, make it extremly long and detailed, use as many as references from Chinese Commnunist Party history or Communism Theory as possible, make it official and academic style, targeting as a report for the central standing committee of the Communist Party of China.",

    "excel_analysis1": "我希望了解给出的excel的整体情况，其中我想知道来自山西省的有哪些人？硕士学历以及更高学历的有哪些人？",
    "meeting_minutes": "我有一个会议的录音文件，我希望生成一份完整详细正规的会议纪要。"
}
request_message = {
    "messages": [
        {
            "role": "user",
            "content": request_dict["meeting_minutes"],
        }
    ],
}

if __name__ == "__main__":
    for event in agent.stream(request_message):
        _DEFAULT_RENDERER.render_stream_event(event)
if __name__ == "__main__":
    example_result = agent.invoke(request_message)
if __name__ == "__main__":
    _DEFAULT_RENDERER.render_final_output(example_result)
    print(example_result['files'])

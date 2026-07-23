#!/usr/bin/env python3
import subprocess
import os

def build():
    with open('README.md', 'r', encoding='utf-8') as f:
        md = f.read()

    proc = subprocess.run(['npx', '-y', 'marked', '--gfm'], input=md, text=True, capture_output=True, check=True)
    body_html = proc.stdout

    template = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compute Resource 세미나</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.8.1/github-markdown.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}, {{left: '\\\\(', right: '\\\\)', display: false}}, {{left: '\\\\[', right: '\\\\]', display: true}}]}});\"></script>
    <style>
        :root {{
            color-scheme: dark;
        }}
        body {{
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 24px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }}
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 1000px;
            margin: 0 auto;
            padding: 45px;
            background-color: #0d1117;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        }}
        .markdown-body img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.4);
            border: 1px solid #30363d;
        }}
        @media (max-width: 767px) {{
            body {{
                padding: 10px;
            }}
            .markdown-body {{
                padding: 20px 15px;
            }}
        }}
    </style>
</head>
<body>
    <article class="markdown-body">
{body_html}
    </article>
</body>
</html>'''

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(template)

    print('Successfully generated index.html!')

if __name__ == '__main__':
    build()

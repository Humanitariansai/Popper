import os
import base64
from typing import List

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def html_img_tag(img_path, width=600):
    ext = os.path.splitext(img_path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        b64 = img_to_base64(img_path)
        return f'<img src="data:image/{ext[1:]};base64,{b64}" width="{width}">' 
    return f'<img src="{img_path}" width="{width}">' 

def html_table_from_dict(d, title=None):
    html = f'<h3>{title}</h3>' if title else ''
    html += '<table border="1" cellpadding="4" cellspacing="0">'
    html += '<tr>' + ''.join(f'<th>{k}</th>' for k in d.keys()) + '</tr>'
    html += '<tr>' + ''.join(f'<td>{v}</td>' for v in d.values()) + '</tr>'
    html += '</table>'
    return html

def html_table_from_nested_dict(d, title=None):
    html = f'<h3>{title}</h3>' if title else ''
    html += '<table border="1" cellpadding="4" cellspacing="0">'
    html += '<tr>' + ''.join(f'<th>{k}</th>' for k in d.keys()) + '</tr>'
    html += '<tr>'
    for v in d.values():
        if isinstance(v, dict):
            html += '<td>' + html_table_from_dict(v) + '</td>'
        else:
            html += f'<td>{v}</td>'
    html += '</tr></table>'
    return html
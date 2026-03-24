from typing import Any
import html

__all__ = ['xml_start_tag', 'xml_end_tag']


def xml_start_tag(tag: str, attributes: dict[str, Any], self_close: bool = False) -> str:
    attributes_str = ''
    if len(attributes) > 0:
        attribute_lines = []
        for key, value in attributes.items():
            if value is None:
                continue
            value_str = str(value)
            value_str = html.escape(value_str, quote=True)
            attributes_str += f'{key}="{value_str}"'
        attributes_str = ' ' + ' '.join(attribute_lines)
    if not self_close:
        return f'<{tag}{attributes_str}>'
    return f'<{tag}{attributes_str}/>'


def xml_end_tag(tag: str) -> str:
    return f'<{tag}>'

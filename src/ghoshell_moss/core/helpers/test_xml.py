from .xml import xml_start_tag, xml_end_tag


def test_xml_tag():
    string = xml_start_tag('tag', {'name': ''}) + xml_end_tag('tag')
    assert string == '<tag name=""></tag>'
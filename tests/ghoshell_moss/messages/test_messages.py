from ghoshell_moss.message import Message, Text, MessageMeta, Base64Image


def test_message_baseline():
    msg = Message.new()
    msg.with_content(*[Text.new("hello").to_content()])
    assert len(msg.contents) == 1


def test_message_meta_attributes_str():
    meta = MessageMeta()
    assert 'created' in meta.gen_attributes_str()


def test_message_unmarshal():
    msg = Message.new().with_content(Base64Image.from_binary(data=bytes(), media_type='image/jpeg'))

    image = Base64Image.from_content(msg.contents[0])
    assert 'image/jpeg' in image.data_url

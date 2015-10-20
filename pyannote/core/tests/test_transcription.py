from pyannote.core import Transcription

def test_creation():
    transcription = Transcription()
    transcription.add_edge(3, 5, speech="hi there, I'm Penny",
                           speaker='Penny')
    transcription.add_edge(5, 5.5)
    transcription.add_edge(5.5, 7, speech="hi. I'm Leonard", speaker='Leonard')

    assert list(transcription.edges_iter(data=True)) == [(3.0, 5.0, {'speech': "hi there, I'm Penny", 'speaker': 'Penny'}),
                                                         (5.0, 5.5, {}),
                                                         (5.5, 7.0, {'speech': "hi. I'm Leonard", 'speaker': 'Leonard'})]

    assert transcription.temporal_sort() == [3.0, 5.0, 5.5, 7.0]

### Version 0.5.1 (2016-02-17)

  - improve: notebook display
  - improve: Annotation.anonymize_{tracks|labels} no longer use Unknown instances
  - improve: empty segments are now printed as "[]"

### Version 0.4.7 (2016-02-04)

  - feat: deterministic order in Annotation.co_iter
  - fix: LabelMatrix.argmax corner case
  - setup: update dependencies

### Version 0.4.4 (2015-11-02)

  - feat: Travis continuous integration

### Version 0.4.3 (2015-10-28)

  - fix: Python 2/3 notebook representations
  - fix: bug in Scores with integer-values segments

### Version 0.4.1 (2015-10-27)

  - fix: update Scores.from_df to pandas 0.17

### Version 0.4 (2015-10-26)

  - feat: Python 3 support
  - feat: pytest test suite
  - fix: Annotation comparison
  - fix: deterministic order in Annotation.itertracks

### Version 0.3.6 (2015-05-06)

  - feat: LabelMatrix save/load methods

### Version 0.3.4 (2015-03-04)

  - fix: MAJOR bug in Annotation lazy-update

### Version 0.3.3 (2015-02-27)

  - fix: Scores IPython display

### Version 0.3.1 (2015-01-26)

  - feat: new Annotation.update method
  - improve: Annotation.subset support for any label iterable

### Version 0.3 (2014-12-04)

  - refactor: rewrote Scores internals
  - setup: use pandas 0.15.1+

### Version 0.2.5 (2014-11-21)

  - setup: use pyannote-banyan 0.1.6

### Version 0.2.4 (2014-11-18)

  - fix: extent of empty Timeline

### Version 0.2.3 (2014-11-14)

  - fix: force revert to pandas 0.13.1 for Scores to work again...

### Version 0.2.2 (2014-11-12)

  - setup: use banyan 0.1.5.1 from GitHub

### Version 0.2.1 (2014-10-30)

  - feat: pyannote/core Docker image
  - feat(Timeline): add from_df constructor

### Version 0.2 (2014-10-24)

  - breaking change: new PyAnnote JSON format

### Version 0.1 (2014-08-05)

  - fix(Transcription): fix potential edge/key conflict during alignment

### Version 0.0.5 (2014-07-23)

  - feat(SlidingWindow): add durationToSamples (and vice-versa)
  - fix(Transcription): fix loading from JSON
  - fix(Transcription): fix cropping corner cases
  - docs: add installation instruction for IPython display support
  - docs(Scores): add IPython documentation for Scores

### Version 0.0.3 (2014-06-02)

  - feat(Annotation): add 'collar' param to .smooth()
  - refactor(Annotation): remove support for >> operator
  - maintain(Mapping): remove label mapping data structure
  - feat(LabelMatrix): add IPython display
  - improve(LabelMatrix): 10x faster cooccurrence matrix
  - feat(Scores): add IPython display
  - feat(Transcription): add edge timerange prediction
  - feat(Transcription): add node temporal sort
  - fix(Transcription): make label_timeline return a copy
  - fix(Transcription): fix IPython display
  - docs(Transcription): add IPython documentation for Transcription

### Version 0.0.2 (2014-05-06)

  - feat: Transcription data structure (annotation graph)

### Version 0.0.1 (2014-05-02)

  - first public version
